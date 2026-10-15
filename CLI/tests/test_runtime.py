from __future__ import annotations

import contextlib
import types
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from CLI.runtime import (GemmaRuntime, RuntimeProblem, model_load_options,
                          needs_resident_cpu_embeddings, plan_text_placement,
                          validate_full_gpu_fit)


class TokenBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, parsed, raw="Hello<turn|>", input_length=10):
        self.parsed = parsed
        self.raw = raw
        self.input_length = input_length
        self.prefix = None
        self.options = None

    def apply_chat_template(self, messages, **kwargs):
        self.options = kwargs
        return "<|turn>model\n"

    def __call__(self, prompt, **kwargs):
        return TokenBatch(input_ids=types.SimpleNamespace(shape=(1, self.input_length)))

    def decode(self, generated, **kwargs):
        return self.raw

    def parse_response(self, raw, prefix):
        self.prefix = prefix
        return self.parsed


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType("transformers")
        module.StoppingCriteria = object
        self.modules = patch.dict("sys.modules", {"transformers": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def runtime(self, parsed, raw="Hello<turn|>", output_length=3, input_length=10):
        runtime = GemmaRuntime(max_new_tokens=20, max_context=100)
        runtime.tokenizer = FakeTokenizer(parsed, raw, input_length)
        runtime.model = Mock(device="cpu")
        runtime.model.generate.return_value = [[0] * (input_length + output_length)]
        runtime.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        runtime.stopping_criteria = list
        return runtime

    def test_generation_passes_native_tools_and_prefix(self):
        runtime = self.runtime({"content": "Hello", "thinking": "private reasoning"})
        turn = runtime.generate([{"role": "user", "content": "Hi"}], [{"name": "read_file"}])
        self.assertEqual(turn.content, "Hello")
        self.assertFalse(hasattr(turn, "thinking"))
        self.assertEqual(runtime.tokenizer.prefix, "<|turn>model\n")
        self.assertEqual(runtime.tokenizer.options["tools"][0]["function"]["name"], "read_file")
        self.assertFalse(runtime.tokenizer.options["enable_thinking"])

    def test_legacy_parser_without_prefix(self):
        runtime = self.runtime({"content": "Hello"})
        runtime.tokenizer.parse_response = lambda raw: {"content": "Hello"}
        self.assertEqual(runtime.generate([], []).content, "Hello")

    def test_unparsed_tool_call_fails_closed(self):
        runtime = self.runtime({"content": ""}, raw="<|tool_call>call:broken{}<tool_call|>")
        with self.assertRaisesRegex(RuntimeProblem, "No action"):
            runtime.generate([], [])

    def test_context_limit_does_not_generate(self):
        runtime = self.runtime({"content": "Hello"}, input_length=90)
        with self.assertRaisesRegex(RuntimeProblem, "context budget"):
            runtime.generate([], [])
        runtime.model.generate.assert_not_called()

    def test_partial_response_is_not_accepted(self):
        runtime = self.runtime({"content": "partial"}, output_length=20)
        with self.assertRaisesRegex(RuntimeProblem, "token limit"):
            runtime.generate([], [])

    def test_deadline_stops_without_accepting_output(self):
        runtime = self.runtime({"content": "partial"})
        with patch("CLI.runtime.time.monotonic", side_effect=[0, 200]):
            with self.assertRaisesRegex(RuntimeProblem, "partial response"):
                runtime.generate([], [])

    def test_bad_parsed_shape_is_rejected(self):
        runtime = self.runtime({"content": [], "tool_calls": "not a list"})
        with self.assertRaisesRegex(RuntimeProblem, "unexpected shape"):
            runtime.generate([], [])

    def fake_torch(self, available=True, capability=(6, 1)):
        return types.SimpleNamespace(
            float16="float16", float32="float32", bfloat16="bfloat16",
            cuda=types.SimpleNamespace(is_available=lambda: available,
                                       device_count=lambda: 1,
                                       get_device_capability=lambda index: capability),
        )

    def test_auto_uses_gpu_and_reserves_generation_memory(self):
        gib = 1024 ** 3
        budgets = {0: 4 * gib, "cpu": 10 * gib}
        options = model_load_options(self.fake_torch(), "auto", budgets)
        self.assertEqual(options["device_map"], "auto")
        self.assertEqual(options["max_memory"], {0: 3 * gib, "cpu": 10 * gib})
        self.assertEqual(budgets[0], 4 * gib)

    def test_pascal_uses_float32_and_eager_attention(self):
        options = model_load_options(self.fake_torch(), "auto")
        self.assertEqual(options["dtype"], "float32")
        self.assertEqual(options["attn_implementation"], "eager")

    def test_modern_gpu_uses_bfloat16(self):
        options = model_load_options(self.fake_torch(capability=(8, 6)), "auto")
        self.assertEqual(options["dtype"], "bfloat16")
        self.assertNotIn("attn_implementation", options)

    def test_auto_without_cuda_still_works_on_cpu(self):
        self.assertEqual(model_load_options(self.fake_torch(available=False), "auto"),
                         {"device_map": "cpu", "dtype": "auto"})

    def test_explicit_cpu_does_not_inspect_cuda(self):
        self.assertEqual(model_load_options(None, "cpu"), {"device_map": "cpu", "dtype": "auto"})

    def test_explicit_cuda_fails_without_gpu(self):
        with self.assertRaisesRegex(RuntimeProblem, "CPU-only"):
            model_load_options(self.fake_torch(available=False), "cuda")

    def test_explicit_cuda_does_not_enable_cpu_offload(self):
        options = model_load_options(self.fake_torch(), "cuda", {0: 4 * 1024 ** 3})
        self.assertEqual(options["device_map"], "cuda")
        self.assertNotIn("max_memory", options)

    def test_execution_summary_reports_mixed_placement(self):
        runtime = GemmaRuntime()
        self.assertEqual(runtime.execution_summary, "not loaded")
        runtime.model = types.SimpleNamespace(hf_device_map={"embedding": 0, "layer": "cpu"})
        self.assertEqual(runtime.execution_summary, "cpu + cuda:0")

    def test_gpu_memory_budget_cannot_be_negative(self):
        options = model_load_options(self.fake_torch(), "auto", {0: 512 * 1024 ** 2, "cpu": 1024 ** 3})
        self.assertEqual(options["max_memory"][0], 0)

    def test_large_per_layer_embedding_stays_resident_on_cpu(self):
        config = types.SimpleNamespace(vocab_size_per_layer_input=262144,
                                       num_hidden_layers=35, hidden_size_per_layer_input=256)
        self.assertTrue(needs_resident_cpu_embeddings(config, {0: 3 * 1024 ** 3}))
        self.assertFalse(needs_resident_cpu_embeddings(config, {0: 8 * 1024 ** 3}))
        self.assertFalse(needs_resident_cpu_embeddings(config, None))

    def test_resident_plan_puts_head_and_fitting_layers_on_gpu(self):
        budgets = {0: 1000, "cpu": 10000}
        placement = plan_text_placement([200, 300, 400], 400, budgets)
        self.assertEqual(placement, {"": "cpu", "lm_head": "cuda:0",
                                     "model.layers.0": "cuda:0", "model.layers.1": "cuda:0"})
        self.assertEqual(budgets[0], 1000)

    def test_resident_plan_supports_more_than_one_gpu(self):
        placement = plan_text_placement([300, 300], 400, {0: 500, 1: 800})
        self.assertEqual(placement["lm_head"], "cuda:0")
        self.assertEqual(placement["model.layers.0"], "cuda:1")
        self.assertEqual(placement["model.layers.1"], "cuda:1")

    def test_resident_plan_never_moves_oversized_modules(self):
        self.assertEqual(plan_text_placement([2000], 3000, {0: 1000}), {"": "cpu"})

    def test_full_gpu_fit_rejects_checkpoint_larger_than_vram(self):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                       get_device_properties=lambda index: types.SimpleNamespace(total_memory=4_000)),
        )
        with patch("CLI.runtime.Path.glob", return_value=[types.SimpleNamespace(stat=lambda: types.SimpleNamespace(st_size=4_000))]):
            with self.assertRaisesRegex(RuntimeProblem, "Full-GPU loading is not possible"):
                validate_full_gpu_fit(Path("model"), torch)


if __name__ == "__main__":
    unittest.main()
