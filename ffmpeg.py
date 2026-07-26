"""Discover a user-installed FFmpeg executable without network access."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys


IS_WINDOWS = sys.platform.startswith("win32")
FFMPEG_OVERRIDE_ENV = "ANKI_AUDIO_NORMALIZER_FFMPEG"


def _startupinfo():
    if not IS_WINDOWS:
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


class FFmpegProvider:
    """Locate and validate FFmpeg supplied by the user or operating system."""

    def __init__(self):
        self.executable = None
        self.version = None
        self.last_error = None

    @staticmethod
    def _system_candidates():
        candidates = [
            os.environ.get(FFMPEG_OVERRIDE_ENV),
            shutil.which("ffmpeg"),
        ]
        if sys.platform == "darwin":
            # GUI apps launched from Finder usually do not inherit the shell PATH.
            candidates += [
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
            ]
        elif sys.platform.startswith("linux"):
            candidates += [
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/snap/bin/ffmpeg",
            ]
        elif IS_WINDOWS:
            local_app_data = os.environ.get("LOCALAPPDATA")
            program_files = os.environ.get("ProgramFiles")
            if local_app_data:
                candidates.append(
                    os.path.join(
                        local_app_data,
                        "Microsoft",
                        "WinGet",
                        "Links",
                        "ffmpeg.exe",
                    )
                )
            if program_files:
                candidates.append(
                    os.path.join(program_files, "ffmpeg", "bin", "ffmpeg.exe")
                )
        # Preserve order while avoiding duplicate probes.
        return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))

    @staticmethod
    def _probe(path, arguments, timeout=10):
        return subprocess.run(
            [path, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=_startupinfo(),
            timeout=timeout,
            check=False,
        )

    @classmethod
    def _version(cls, path):
        try:
            completed = cls._probe(path, ["-hide_banner", "-version"])
        except (OSError, subprocess.TimeoutExpired):
            return None
        output = (completed.stdout + completed.stderr).decode(
            "utf-8", errors="replace"
        )
        if completed.returncode != 0:
            return None
        match = re.search(r"^ffmpeg version\s+([^\s]+)", output, re.MULTILINE)
        return match.group(1) if match else "unknown"

    @classmethod
    def _is_usable(cls, path):
        if not path or not os.path.isfile(path):
            return False
        try:
            probes = (
                (["-hide_banner", "-filters"], ("loudnorm",)),
                (
                    ["-hide_banner", "-encoders"],
                    ("libmp3lame", "libopus", "aac", "flac", "pcm_s16le"),
                ),
            )
            for arguments, required in probes:
                completed = cls._probe(path, arguments)
                output = (completed.stdout + completed.stderr).decode(
                    "utf-8", errors="replace"
                )
                if completed.returncode != 0 or not all(
                    re.search(rf"\b{re.escape(name)}\b", output)
                    for name in required
                ):
                    return False
            return cls._version(path) is not None
        except (OSError, subprocess.TimeoutExpired):
            return False

    def discover(self):
        """Return a usable executable without downloading or modifying anything."""

        self.executable = None
        self.version = None
        self.last_error = None
        for candidate in self._system_candidates():
            if self._is_usable(candidate):
                self.executable = os.path.realpath(candidate)
                self.version = self._version(self.executable)
                return self.executable
        return None

    def ensure_available(self, progress_callback=None):
        """Compatibility shim: discovery is deliberately network-free."""

        del progress_callback
        existing = self.discover()
        if existing:
            return existing
        self.last_error = (
            "Install FFmpeg with the loudnorm, MP3, Opus, AAC, FLAC, and WAV "
            "encoders, restart Anki, and try again. The add-on does not download "
            "or execute third-party binaries automatically."
        )
        return None


ffmpeg_provider = FFmpegProvider()
