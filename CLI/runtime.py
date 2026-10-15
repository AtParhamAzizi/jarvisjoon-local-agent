"""Offline text inference using the Gemma checkpoint already in this project."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "gemma-4-E2B-it"


class RuntimeProblem(RuntimeError):
    """A model setup or inference problem that can be explained to the user."""


@dataclass
class AssistantTurn:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def inspect_hardware() -> dict[str, Any]:
    try:
        import torch

        available = torch.cuda.is_available()
        devices = []
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({"index": index, "name": properties.name,
                                "memory_gib": round(properties.total_memory / 1024 ** 3, 2)})
        return {"torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
                "cuda_available": available, "gpus": devices}
    except Exception as error:
        return {"cuda_available": False, "gpus": [], "error": str(error)}


def model_load_options(torch, device: str, memory_limits: dict | None = None) -> dict:
    if device == "cpu":
        return {"device_map": "cpu", "dtype": "auto"}
    if not torch.cuda.is_available():
        if device == "cuda":
            raise RuntimeProblem("CUDA is unavailable. Check CLI doctor: a CPU-only PyTorch build "
                                 "must be replaced with a compatible CUDA build, and the NVIDIA driver must work.")
        return {"device_map": "cpu", "dtype": "auto"}
    native_bf16 = all(torch.cuda.get_device_capability(index)[0] >= 8
                      for index in range(torch.cuda.device_count()))
    options = {"device_map": "auto" if device == "auto" else "cuda",
               "dtype": torch.bfloat16 if native_bf16 else torch.float32}
    if not native_bf16:
        options["attn_implementation"] = "eager"
    if device == "auto" and memory_limits is not None:
        options["max_memory"] = {
            target: max(0, budget - 1024 ** 3) if isinstance(target, int) else budget
            for target, budget in memory_limits.items()
        }
    return options


def validate_full_gpu_fit(model_path: Path, torch) -> None:
    if not torch.cuda.is_available():
        raise RuntimeProblem("CUDA is unavailable in this Python environment. Install the GPU requirements and verify the NVIDIA driver.")
    gpu_bytes = sum(torch.cuda.get_device_properties(index).total_memory
                    for index in range(torch.cuda.device_count()))
    weight_bytes = sum(path.stat().st_size for path in model_path.glob("*.safetensors"))
    required = int(weight_bytes * 1.15)
    if required > gpu_bytes:
        raise RuntimeProblem(
            f"Full-GPU loading is not possible: the checkpoint needs about {required / 1024**3:.1f} GiB "
            f"including runtime headroom, but available GPUs provide {gpu_bytes / 1024**3:.1f} GiB. "
            "Use --device auto for GPU/CPU offload or choose a smaller/quantized model with --model."
        )


def needs_resident_cpu_embeddings(config, memory_limits: dict | None) -> bool:
    if not memory_limits:
        return False
    gpu_budget = max((budget for target, budget in memory_limits.items() if isinstance(target, int)), default=0)
    embedding_bytes = (config.vocab_size_per_layer_input * config.num_hidden_layers
                       * config.hidden_size_per_layer_input * 2)
    return embedding_bytes > gpu_budget


def plan_text_placement(layer_sizes: list[int], head_size: int, memory_limits: dict) -> dict:
    remaining = {target: budget for target, budget in memory_limits.items() if isinstance(target, int)}
    placement = {"": "cpu"}
    modules = [("lm_head", head_size)] + [(f"model.layers.{index}", size)
                                         for index, size in enumerate(layer_sizes)]
    for name, size in modules:
        for target, budget in remaining.items():
            if size <= budget:
                placement[name] = f"cuda:{target}"
                remaining[target] -= size
                break
    return placement


def place_resident_text_modules(model, torch, memory_limits: dict, gpu_dtype) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import recursively_apply

    cpu_dtype = model.model.embed_tokens.weight.dtype

    class PrecisionAlignHook(AlignDevicesHook):
        def pre_forward(self, module, *args, **kwargs):
            self.shared_states = kwargs.get("shared_kv_states")
            args, kwargs = super().pre_forward(module, *args, **kwargs)

            def cast_tensor(tensor):
                return tensor.to(dtype=gpu_dtype) if tensor.is_floating_point() else tensor

            args = recursively_apply(cast_tensor, args)
            kwargs = {key: value if key in {"past_key_values", "shared_kv_states"}
                      else recursively_apply(cast_tensor, value) for key, value in kwargs.items()}
            if self.shared_states is not None:
                for key, values in self.shared_states.items():
                    self.shared_states[key] = tuple(tensor.to(device=self.execution_device, dtype=gpu_dtype)
                                                    for tensor in values)
            return args, kwargs

        def post_forward(self, module, output):
            if self.shared_states is not None:
                for key, values in self.shared_states.items():
                    self.shared_states[key] = tuple(tensor.to(device="cpu", dtype=cpu_dtype) for tensor in values)
            output = recursively_apply(lambda tensor: tensor.to(dtype=cpu_dtype)
                                       if tensor.is_floating_point() else tensor, output)
            return super().post_forward(module, output)

    def module_bytes(module):
        element_size = torch.empty((), dtype=gpu_dtype).element_size()
        return sum(tensor.numel() * element_size
                   for tensor in (*module.parameters(), *module.buffers()))

    placement = plan_text_placement([module_bytes(layer) for layer in model.model.layers],
                                    module_bytes(model.lm_head), memory_limits)
    for name, target in placement.items():
        if not name:
            continue
        module = model.get_submodule(name)
        if name == "lm_head":
            module.weight = torch.nn.Parameter(module.weight.detach().to(device=target, dtype=gpu_dtype),
                                                requires_grad=False)
        else:
            module.to(device=target, dtype=gpu_dtype)
        add_hook_to_module(module, PrecisionAlignHook(
            execution_device=target, io_same_device=True,
            skip_keys=["past_key_values", "shared_kv_states"],
        ))
    model.hf_device_map = placement


def inspect_model(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    missing = [name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
               if not (model_path / name).is_file()]
    templates = [model_path / "chat_template.jinja", model_path / "chat_template.jinja.txt"]
    if not any(template.is_file() for template in templates):
        missing.append("chat_template.jinja (or chat_template.jinja.txt)")
    weights = list(model_path.glob("*.safetensors"))
    if not weights:
        missing.append("Safetensors model weights")
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            missing.extend(name for name in shards if not (model_path / name).is_file())
        except (ValueError, KeyError, TypeError):
            missing.append("a valid model.safetensors.index.json")
    return {
        "model": str(model_path),
        "missing": missing,
        "weight_bytes": sum(weight.stat().st_size for weight in weights),
        "dependencies": {name: importlib.util.find_spec(name) is not None
                         for name in ("torch", "transformers", "accelerate")},
        "note": "File checks only: this does not prove that the weights load or fit in memory.",
    }


class GemmaRuntime:
    def __init__(self, model_path: Path = DEFAULT_MODEL, device: str = "auto",
                 max_context: int = 4096, max_new_tokens: int = 768,
                 generation_timeout: int = 120):
        self.model_path = model_path.resolve()
        self.device = device
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.generation_timeout = generation_timeout
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.stopping_criteria = None
        self.precision = "auto"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def execution_summary(self) -> str:
        if not self.loaded:
            return "not loaded"
        mapping = getattr(self.model, "hf_device_map", None)
        if mapping:
            devices = sorted({f"cuda:{target}" if isinstance(target, int) else str(target)
                              for target in mapping.values()})
            return " + ".join(devices)
        return str(self.model.device)

    def load(self) -> None:
        if self.loaded:
            return
        report = inspect_model(self.model_path)
        if report["missing"]:
            raise RuntimeProblem("The local checkpoint is incomplete: " + ", ".join(report["missing"]))
        if not all(report["dependencies"].values()):
            raise RuntimeProblem(
                "The model libraries aren't installed yet. Run: python -m pip install -r CLI/requirements.txt"
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import torch
            from accelerate.utils import get_max_memory
            from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig, StoppingCriteriaList

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
            )
            if not tokenizer.chat_template:
                for name in ("chat_template.jinja", "chat_template.jinja.txt"):
                    template = self.model_path / name
                    if template.is_file():
                        tokenizer.chat_template = template.read_text(encoding="utf-8")
                        break
            if not callable(getattr(tokenizer, "parse_response", None)):
                raise RuntimeProblem("This Transformers version lacks response parsing. Install CLI/requirements.txt.")
            memory_limits = get_max_memory() if self.device == "auto" and torch.cuda.is_available() else None
            if self.device == "cuda":
                validate_full_gpu_fit(self.model_path, torch)
            load_options = model_load_options(torch, self.device, memory_limits)
            config = Gemma4TextConfig.from_pretrained(str(self.model_path), local_files_only=True)
            gpu_dtype = load_options["dtype"]
            legacy_gpu = gpu_dtype == torch.float32
            resident_embeddings = self.device == "auto" and (legacy_gpu or needs_resident_cpu_embeddings(
                config, load_options.get("max_memory")))
            gpu_memory_limits = load_options.get("max_memory")
            if resident_embeddings:
                load_options["device_map"] = "cpu"
                load_options.pop("max_memory", None)
                load_options["dtype"] = torch.bfloat16

            class TextCheckpointModel(Gemma4ForCausalLM):
                _keys_to_ignore_on_load_unexpected = [r"model\.(audio_tower|vision_tower|embed_audio|embed_vision)\."]

            model, loading_info = TextCheckpointModel.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
                config=config, key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True,
                **load_options,
            )
            if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
                raise RuntimeProblem("The text checkpoint did not load completely. No inference was attempted.")
            if resident_embeddings:
                place_resident_text_modules(model, torch, gpu_memory_limits, gpu_dtype)
            model.eval()
            self.torch = torch
            self.tokenizer = tokenizer
            self.model = model
            self.stopping_criteria = StoppingCriteriaList
            self.precision = (f"CPU {load_options['dtype']} / GPU {gpu_dtype}" if resident_embeddings
                              else str(load_options["dtype"]))
        except RuntimeProblem:
            raise
        except Exception as error:
            raise RuntimeProblem(
                "I couldn't load the local Gemma checkpoint. Check compatible libraries and available RAM/VRAM. "
                "For a model larger than GPU memory, use --device auto for GPU/CPU offloading instead of --device cuda. "
                f"Details: {type(error).__name__}: {error}"
            ) from error

    def generate(self, messages: list[dict], definitions: list[dict]) -> AssistantTurn:
        self.load()
        from transformers import StoppingCriteria

        deadline = time.monotonic() + self.generation_timeout

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        tools = [{"type": "function", "function": definition} for definition in definitions]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=tools or None, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_length = inputs["input_ids"].shape[-1]
            if input_length + self.max_new_tokens > self.max_context:
                raise RuntimeProblem(
                    "This conversation is too large for the context budget. Use /reset, ask a smaller question, "
                    "or increase --context-tokens if your hardware has room. Nothing was silently truncated."
                )
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    stopping_criteria=self.stopping_criteria([Deadline()]),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = outputs[0][input_length:]
            if time.monotonic() >= deadline:
                raise RuntimeProblem("Generation took too long. I stopped without executing its partial response.")
            if len(generated) >= self.max_new_tokens:
                raise RuntimeProblem("The reply hit its token limit. Try a smaller task or increase --max-new-tokens.")
            raw = self.tokenizer.decode(generated, skip_special_tokens=False)
            parser = self.tokenizer.parse_response
            parse_options = {}
            if "prefix" in inspect.signature(parser).parameters:
                parse_options["prefix"] = prompt
            parsed = parser(raw, **parse_options)
            if not isinstance(parsed, dict):
                raise RuntimeProblem("Gemma returned a response I couldn't safely interpret.")
            calls = parsed.get("tool_calls") or []
            if "<|tool_call>" in raw and not calls:
                raise RuntimeProblem("A tool call wasn't parsed correctly. No action was taken.")
            content = parsed.get("content") or ""
            if not isinstance(content, str) or not isinstance(calls, list):
                raise RuntimeProblem("Gemma's response had an unexpected shape. No action was taken.")
            return AssistantTurn(content=content, tool_calls=calls)
        except RuntimeProblem:
            raise
        except Exception as error:
            raise RuntimeProblem(f"Local inference stopped: {type(error).__name__}: {error}") from error
