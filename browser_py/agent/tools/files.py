"""File management tool — sandboxed to the user's workspace directory.

All paths are resolved relative to the workspace. Escape attempts
(../../etc/passwd) are blocked.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from browser_py.agent.config import get_workspace

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "files",
        "description": (
            "Manage files in the workspace directory. Read, write, list, move, "
            "copy, and delete files. All paths are relative to the workspace — "
            "you cannot access files outside it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "move", "copy", "delete", "mkdir", "info"],
                    "description": (
                        "File action:\n"
                        "- read: Read file contents (requires 'path')\n"
                        "- write: Write/create file (requires 'path' and 'content')\n"
                        "- list: List directory contents (optional 'path', default root)\n"
                        "- move: Move/rename file (requires 'path' and 'destination')\n"
                        "- copy: Copy file (requires 'path' and 'destination')\n"
                        "- delete: Delete file or directory (requires 'path')\n"
                        "- mkdir: Create directory (requires 'path')\n"
                        "- info: Get file size, type, modification time (requires 'path')"
                    ),
                },
                "path": {"type": "string", "description": "File or directory path (relative to workspace)"},
                "content": {"type": "string", "description": "File content for write action"},
                "destination": {"type": "string", "description": "Destination path for move/copy"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)"},
            },
            "required": ["action"],
        },
    },
}


class FilesTool:
    """Sandboxed file operations."""

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace

    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = get_workspace()
        self._workspace = self._workspace.resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        return self._workspace

    def _resolve(self, path: str) -> Path:
        """Resolve a path within the workspace. Blocks escapes."""
        resolved = (self.workspace / path).resolve()
        ws_resolved = self.workspace.resolve()
        if not str(resolved).startswith(str(ws_resolved)):
            raise PermissionError(f"Access denied: path escapes workspace — {path}")
        return resolved

    def execute(self, **params: Any) -> str:
        action = params.get("action", "")

        try:
            if action == "read":
                return self._read(params)
            elif action == "write":
                return self._write(params)
            elif action == "list":
                return self._list(params)
            elif action == "move":
                return self._move(params)
            elif action == "copy":
                return self._copy(params)
            elif action == "delete":
                return self._delete(params)
            elif action == "mkdir":
                return self._mkdir(params)
            elif action == "info":
                return self._info(params)
            else:
                return f"Unknown action: {action}"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            return f"Error: {e}"

    def _read(self, params: dict) -> str:
        path = params.get("path", "")
        if not path:
            return "Error: 'path' required"
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"File not found: {path}"
        if resolved.is_dir():
            return f"'{path}' is a directory. Use action='list' instead."
        encoding = params.get("encoding", "utf-8")
        try:
            content = resolved.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Binary file — return size info
            size = resolved.stat().st_size
            return f"Binary file ({size} bytes). Cannot read as text."
        # Cap at 50KB
        if len(content) > 50_000:
            content = content[:50_000] + f"\n\n... (truncated, {len(content)} chars total)"
        return content

    def _write(self, params: dict) -> str:
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return "Error: 'path' required"
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        encoding = params.get("encoding", "utf-8")
        resolved.write_text(content, encoding=encoding)
        return f"Written: {path} ({len(content)} chars)"

    def _list(self, params: dict) -> str:
        path = params.get("path", ".")
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Directory not found: {path}"
        if not resolved.is_dir():
            return f"'{path}' is a file, not a directory."

        entries = sorted(resolved.iterdir())
        if not entries:
            return f"(empty directory: {path})"

        lines = []
        for entry in entries:
            rel = entry.relative_to(self.workspace)
            if entry.is_dir():
                lines.append(f"  📁 {rel}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                lines.append(f"  📄 {rel} ({size_str})")
        return "\n".join(lines)

    def _move(self, params: dict) -> str:
        src = params.get("path", "")
        dst = params.get("destination", "")
        if not src or not dst:
            return "Error: 'path' and 'destination' required"
        resolved_src = self._resolve(src)
        resolved_dst = self._resolve(dst)
        if not resolved_src.exists():
            return f"Source not found: {src}"
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved_src), str(resolved_dst))
        return f"Moved: {src} → {dst}"

    def _copy(self, params: dict) -> str:
        src = params.get("path", "")
        dst = params.get("destination", "")
        if not src or not dst:
            return "Error: 'path' and 'destination' required"
        resolved_src = self._resolve(src)
        resolved_dst = self._resolve(dst)
        if not resolved_src.exists():
            return f"Source not found: {src}"
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        if resolved_src.is_dir():
            shutil.copytree(str(resolved_src), str(resolved_dst))
        else:
            shutil.copy2(str(resolved_src), str(resolved_dst))
        return f"Copied: {src} → {dst}"

    def _delete(self, params: dict) -> str:
        path = params.get("path", "")
        if not path:
            return "Error: 'path' required"
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Not found: {path}"
        if resolved == self.workspace:
            return "Error: cannot delete the workspace root"
        if resolved.is_dir():
            shutil.rmtree(resolved)
            return f"Deleted directory: {path}"
        else:
            resolved.unlink()
            return f"Deleted: {path}"

    def _mkdir(self, params: dict) -> str:
        path = params.get("path", "")
        if not path:
            return "Error: 'path' required"
        resolved = self._resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {path}"

    def _info(self, params: dict) -> str:
        path = params.get("path", "")
        if not path:
            return "Error: 'path' required"
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Not found: {path}"
        stat = resolved.stat()
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        kind = "directory" if resolved.is_dir() else resolved.suffix or "file"
        size = stat.st_size
        return f"Path: {path}\nType: {kind}\nSize: {size} bytes\nModified: {mtime}"
