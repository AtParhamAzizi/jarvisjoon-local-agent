"""Permission checks around the existing workspace tools."""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

from tools.build_tools import build_tools
from tools.workspace import Workspace


READ_TOOLS = {"list_directory", "read_file", "search_files", "find_files", "git_status"}
EDIT_TOOLS = {"write_file", "edit_file"}
PROTECTED_PARTS = {
    ".git", ".env", ".ssh", ".aws", ".azure", ".gnupg", ".agents", ".gapcode",
    "models", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
}
MAX_FILE_BYTES = 1_048_576
MAX_SCAN_FILES = 10_000
MAX_RESULT_CHARS = 12_000


def failure(message: str, error_type: str = "permission_denied") -> dict:
    return {"ok": False, "error_type": error_type, "message": message}


class SafeWorkspace(Workspace):
    def resolve(self, path: str) -> Path:
        lexical = Path(path)
        target = super().resolve(path)
        relative = target.relative_to(self.root)
        for part in (*lexical.parts, *relative.parts):
            lowered = part.lower()
            if (lowered in PROTECTED_PARTS or lowered.startswith(".env.")
                    or lowered.endswith((".pem", ".key", ".p12", ".pfx", ".safetensors"))
                    or lowered in {"credentials", "credentials.json", "id_rsa", "id_ed25519"}):
                raise PermissionError("That path is protected. Choose a regular project file instead.")
        lexical_target = Path(os.path.abspath(self.root / lexical))
        lexical_relative = lexical_target.relative_to(self.root)
        if ".." in lexical.parts:
            raise PermissionError("Parent-directory traversal isn't available to this agent.")
        current = self.root
        for part in lexical_relative.parts:
            if (":" in part or part.endswith((".", " "))
                    or part.split(".")[0].lower() in {"con", "prn", "aux", "nul", "com1", "lpt1"}):
                raise PermissionError("That filename isn't available to this agent.")
            current = current / part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                raise PermissionError("Links and junctions aren't available to this agent.")
        return target


class ToolGateway:
    def __init__(self, root: Path, allow_edits: bool = False,
                 approve: Callable[[str], bool] | None = None):
        self.workspace = SafeWorkspace(str(root))
        if not self.workspace.root.is_dir():
            raise ValueError("The workspace must be an existing folder.")
        self.registry = build_tools(self.workspace)
        self.approve = approve or (lambda preview: False)
        self.set_edit_mode(allow_edits)

    def set_edit_mode(self, allow_edits: bool) -> None:
        self.allow_edits = allow_edits
        allowed = READ_TOOLS | (EDIT_TOOLS if allow_edits else set())
        self.schemas = {}
        for definition in self.registry.definitions():
            if definition["name"] in allowed:
                definition["parameters"]["additionalProperties"] = False
                self.schemas[definition["name"]] = definition

    def definitions(self) -> list[dict]:
        return list(self.schemas.values())

    def validate(self, name: str, arguments: dict) -> None:
        if name not in self.schemas:
            if name in EDIT_TOOLS and not self.allow_edits:
                raise PermissionError("File changes are disabled. Enter /build or restart with --allow-edits.")
            raise PermissionError(f"Tool '{name}' isn't enabled. Commands are never enabled here.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        schema = self.schemas[name]["parameters"]
        if set(arguments) - set(schema["properties"]):
            raise ValueError("The tool received unknown arguments.")
        if set(schema["required"]) - set(arguments):
            raise ValueError("The tool is missing required arguments.")
        expected_types = {"string": str, "integer": int, "boolean": bool}
        for key, value in arguments.items():
            expected = expected_types[schema["properties"][key]["type"]]
            if type(value) is not expected:
                raise ValueError(f"'{key}' must be a {schema['properties'][key]['type']}.")
            if isinstance(value, str) and len(value.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("A tool argument is too large.")
        if "max_results" in arguments and not 1 <= arguments["max_results"] <= 200:
            raise ValueError("max_results must be between 1 and 200.")
        if name in {"read_file", "write_file", "edit_file"}:
            target = self.workspace.resolve(arguments["path"])
            if target.exists() and target.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("That file is too large. The current limit is 1 MiB.")

    def execute(self, name: str, arguments: dict) -> dict:
        try:
            self.validate(name, arguments)
            if name in EDIT_TOOLS:
                return self._edit(name, arguments)
            if name == "list_directory":
                directory = self.workspace.resolve(arguments.get("path", "."))
                entries = []
                for target in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
                    try:
                        self.workspace.resolve(str(target))
                    except PermissionError:
                        continue
                    entries.append({"name": target.name, "type": "directory" if target.is_dir() else "file"})
                    if len(entries) >= 200:
                        break
                return {"ok": True, "entries": entries, "limit": 200}
            if name in {"search_files", "find_files"}:
                return self._search(name, arguments)
            if name == "git_status":
                return self._git_status()
            return self.registry.execute(name, arguments).to_dict()
        except (OSError, ValueError, PermissionError, UnicodeError) as error:
            return failure(str(error), type(error).__name__)

    def _git_status(self) -> dict:
        environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            result = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "status", "--short"],
                cwd=self.workspace.root, env=environment, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return failure(str(error), "git_error")
        return {"ok": result.returncode == 0, "stdout": result.stdout[:MAX_RESULT_CHARS],
                "stderr": result.stderr[:MAX_RESULT_CHARS], "exit_code": result.returncode}

    def _candidates(self, target: Path):
        if target.is_file():
            yield target
            return
        for directory, folders, files in os.walk(target, followlinks=False):
            safe_folders = []
            for folder in sorted(folders):
                try:
                    self.workspace.resolve(str(Path(directory) / folder))
                    safe_folders.append(folder)
                except (PermissionError, OSError):
                    continue
            folders[:] = safe_folders
            for filename in sorted(files):
                try:
                    yield self.workspace.resolve(str(Path(directory) / filename))
                except (PermissionError, OSError):
                    continue

    def _search(self, name: str, arguments: dict) -> dict:
        target = self.workspace.resolve(arguments.get("path", "."))
        if not target.exists():
            return failure("That search path doesn't exist.", "not_found")
        query = arguments.get("query", arguments.get("pattern", ""))
        if not query:
            return failure("Please provide a nonempty search.", "invalid_query")
        limit = arguments.get("max_results", 50)
        results = []
        scanned = 0
        truncated = False
        sensitive = arguments.get("case_sensitive", True)
        for candidate in self._candidates(target):
            scanned += 1
            if scanned > MAX_SCAN_FILES or len(results) >= limit:
                truncated = True
                break
            relative = str(candidate.relative_to(self.workspace.root))
            if name == "find_files":
                if fnmatch.fnmatch(candidate.name, query):
                    results.append(relative)
                continue
            try:
                if candidate.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = candidate.read_text(encoding="utf-8")
                if "\x00" in text:
                    continue
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if (query if sensitive else query.lower()) in (line if sensitive else line.lower()):
                    results.append({"file": relative, "line": line_number, "text": line[:1000]})
                    if len(results) >= limit:
                        truncated = True
                        break
        key = "files" if name == "find_files" else "matches"
        return {"ok": True, key: results, "count": len(results), "truncated": truncated}

    def _edit(self, name: str, arguments: dict) -> dict:
        path = self.workspace.resolve(arguments["path"])
        previous = path.read_bytes() if path.exists() else None
        before = previous.decode("utf-8") if previous is not None else ""
        if name == "edit_file":
            old_text = arguments["old_text"]
            if previous is None or not old_text or before.count(old_text) != 1:
                return failure("The old text must occur exactly once in an existing file.", "ambiguous_edit")
            after = before.replace(old_text, arguments["new_text"], 1)
        else:
            after = arguments["content"]
        if len(after.encode("utf-8")) > MAX_FILE_BYTES:
            return failure("The resulting file would exceed 1 MiB.", "size_limit")
        if previous is not None and before == after:
            return {"ok": True, "path": arguments["path"], "changed": False}
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"before/{arguments['path']}", tofile=f"after/{arguments['path']}",
        ))
        preview = f"I'd like to {'update' if previous is not None else 'create'} {arguments['path']}.\n\n{diff}"
        if len(preview) > 30_000:
            return failure("The change is too large to preview safely. Please request a smaller edit.", "preview_limit")
        if not self.approve(preview):
            return failure("You declined this change. The file was not touched.", "approval_denied")
        checked_path = self.workspace.resolve(arguments["path"])
        current = checked_path.read_bytes() if checked_path.exists() else None
        if checked_path != path or current != previous:
            return failure("The file changed after the preview. I haven't overwritten it.", "stale_file")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".jarvisjoon-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(after.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            if previous is not None:
                os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return {"ok": True, "path": arguments["path"], "changed": True, "bytes_written": len(after.encode("utf-8"))}


def bounded_result(result: dict) -> dict:
    encoded = json.dumps(result, ensure_ascii=True)
    if len(encoded) <= MAX_RESULT_CHARS:
        return result
    return {"ok": result.get("ok", False), "truncated": True,
            "message": "Output shortened. Use a narrower search or read a smaller line range.",
            "excerpt": encoded[:MAX_RESULT_CHARS]}
