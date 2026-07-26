"""Small local cache that avoids re-measuring unchanged media."""

import json
import os
import tempfile
from typing import Optional


STATE_VERSION = 1


class ProcessingState:
    def __init__(self, path: str):
        self.path = path
        self.entries = {}
        self._dirty = False
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as state_file:
                payload = json.load(state_file)
            if payload.get("version") == STATE_VERSION:
                entries = payload.get("entries", {})
                self.entries = entries if isinstance(entries, dict) else {}
        except (OSError, ValueError, TypeError):
            self.entries = {}

    def status_for(
        self, filename: str, source_hash: str, recipe_token: str
    ) -> Optional[str]:
        entry = self.entries.get(filename)
        if (
            isinstance(entry, dict)
            and entry.get("source_hash") == source_hash
            and entry.get("recipe") == recipe_token
        ):
            return entry.get("status")
        return None

    def unchanged_file(
        self, filename: str, path: str, recipe_token: str
    ) -> Optional[tuple[str, str]]:
        """Return the cached hash and status when inexpensive metadata matches."""
        entry = self.entries.get(filename)
        if not isinstance(entry, dict) or entry.get("recipe") != recipe_token:
            return None
        try:
            metadata = os.stat(path)
        except OSError:
            return None
        source_hash = entry.get("source_hash")
        status = entry.get("status")
        if (
            isinstance(source_hash, str)
            and isinstance(status, str)
            and entry.get("size") == metadata.st_size
            and entry.get("mtime_ns") == metadata.st_mtime_ns
        ):
            return source_hash, status
        return None

    def origin_for(self, filename: str) -> Optional[str]:
        entry = self.entries.get(filename)
        if isinstance(entry, dict):
            origin = entry.get("origin")
            return origin if isinstance(origin, str) and origin else None
        return None

    def remember(
        self,
        filename: str,
        source_hash: str,
        recipe_token: str,
        status: str,
        origin: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> None:
        entry = {
            "source_hash": source_hash,
            "recipe": recipe_token,
            "status": status,
        }
        if source_path:
            try:
                metadata = os.stat(source_path)
                entry["size"] = metadata.st_size
                entry["mtime_ns"] = metadata.st_mtime_ns
            except OSError:
                pass
        if origin or self.origin_for(filename):
            entry["origin"] = origin or self.origin_for(filename)
        self.entries[filename] = entry
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".audio_state_", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
                json.dump(
                    {"version": STATE_VERSION, "entries": self.entries},
                    state_file,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            os.replace(temporary_path, self.path)
            self._dirty = False
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
