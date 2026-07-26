"""FFmpeg adapter for measuring and transforming audio files."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from typing import Optional

from .core import AudioMeasurement, ProcessingRecipe


class FFmpegError(RuntimeError):
    pass


def _startupinfo():
    if not os.name == "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _loudnorm_json(stderr: str) -> dict:
    candidates = re.findall(
        r"\{\s*\"input_i\"\s*:.*?\"target_offset\"\s*:\s*\"[^\"]+\"\s*\}",
        stderr,
        flags=re.DOTALL,
    )
    if not candidates:
        raise FFmpegError("FFmpeg did not return loudness measurements.")
    try:
        return json.loads(candidates[-1])
    except json.JSONDecodeError as error:
        raise FFmpegError("FFmpeg returned invalid loudness measurements.") from error


class FFmpegBackend:
    def __init__(self, executable: str, timeout_seconds: int = 180):
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def _run(self, command):
        try:
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=_startupinfo(),
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise FFmpegError("FFmpeg timed out while processing audio.") from error
        except OSError as error:
            raise FFmpegError(f"Unable to start FFmpeg: {error}") from error

    def _analyze_once(
        self,
        input_path: str,
        recipe: ProcessingRecipe,
        measurement_loops: int = 1,
    ) -> AudioMeasurement:
        command = [
            self.executable,
            "-hide_banner",
            "-nostats",
        ]
        if measurement_loops > 1:
            command += ["-stream_loop", str(measurement_loops - 1)]
        command += [
            "-i",
            input_path,
            "-map",
            "0:a:0",
            "-af",
            (
                f"loudnorm=I={recipe.target_lufs}:"
                f"TP={recipe.true_peak_dbtp}:LRA=7:"
                "dual_mono=true:print_format=json"
            ),
            "-f",
            "null",
            "-",
        ]
        completed = self._run(command)
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            message = stderr.strip().splitlines()[-1:] or ["unknown FFmpeg error"]
            raise FFmpegError(message[0])
        values = _loudnorm_json(stderr)
        return AudioMeasurement(
            integrated_lufs=_number(values.get("input_i")),
            true_peak_dbtp=_number(values.get("input_tp")),
        )

    def analyze(self, input_path: str, recipe: ProcessingRecipe) -> AudioMeasurement:
        recipe.validate()
        measurement = self._analyze_once(input_path, recipe)
        if (
            measurement.integrated_lufs is None
            and measurement.true_peak_dbtp is not None
        ):
            # EBU R128 needs roughly 400 ms of programme material. Repeat only
            # for measurement so short word clips are handled without padding
            # or changing the saved audio.
            try:
                measurement = self._analyze_once(
                    input_path,
                    recipe,
                    measurement_loops=8,
                )
            except FFmpegError:
                pass
        return measurement

    def transform(
        self,
        input_path: str,
        output_path: str,
        gain_db: float,
        recipe: ProcessingRecipe,
        output_format: Optional[str] = None,
    ) -> None:
        recipe.validate()
        output_format = (output_format or recipe.output_format).lower().lstrip(".")
        profiles = {
            "mp3": (["-codec:a", "libmp3lame", "-q:a", "4"], "mp3"),
            "opus": (["-codec:a", "libopus", "-b:a", "48k"], "opus"),
            "ogg": (["-codec:a", "libopus", "-b:a", "48k"], "ogg"),
            "oga": (["-codec:a", "libopus", "-b:a", "48k"], "ogg"),
            "wav": (["-codec:a", "pcm_s16le"], "wav"),
            "flac": (["-codec:a", "flac"], "flac"),
            "m4a": (["-codec:a", "aac", "-b:a", "96k"], "ipod"),
            "aac": (["-codec:a", "aac", "-b:a", "96k"], "adts"),
            "webm": (["-codec:a", "libopus", "-b:a", "48k"], "webm"),
        }
        if output_format not in profiles:
            raise FFmpegError(
                f'Re-encoding ".{output_format}" is not supported. '
                "Choose MP3 output or leave this file unchanged."
            )
        codec_arguments, muxer = profiles[output_format]
        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".audio_normalizer_",
            suffix=f".{output_format}",
            dir=output_directory,
        )
        os.close(descriptor)
        try:
            command = [
                self.executable,
                "-y",
                "-hide_banner",
                "-nostats",
                "-i",
                input_path,
                "-map",
                "0:a:0",
                "-vn",
                "-map_metadata",
                "-1",
            ]
            if abs(gain_db) > 0.001:
                command += ["-af", f"volume={gain_db:.3f}dB"]
            command += codec_arguments + ["-f", muxer, temporary_path]
            completed = self._run(command)
            if completed.returncode != 0 or os.path.getsize(temporary_path) == 0:
                stderr = completed.stderr.decode("utf-8", errors="replace")
                message = stderr.strip().splitlines()[-1:] or ["unknown FFmpeg error"]
                raise FFmpegError(message[0])
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
