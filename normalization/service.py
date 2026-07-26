"""Application service for idempotent batch audio normalization."""

from dataclasses import dataclass, field
import hashlib
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .core import (
    AudioMeasurement,
    GainDecision,
    LoudnessStatistics,
    ProcessingRecipe,
    decide_gain,
    is_processed_for,
    loudness_statistics,
    marker_info,
    output_extension,
    output_filename,
    strip_processing_marker,
)
from .ffmpeg_backend import FFmpegBackend, FFmpegError


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class FileResult:
    source_filename: str
    status: str
    detail: str
    output_filename: Optional[str] = None
    measurement: Optional[AudioMeasurement] = None
    output_measurement: Optional[AudioMeasurement] = None
    decision: Optional[GainDecision] = None
    format_converted: bool = False


@dataclass
class BatchResult:
    files: List[FileResult] = field(default_factory=list)
    cancelled: bool = False
    target_lufs: Optional[float] = None

    @property
    def replacements(self) -> Dict[str, str]:
        return {
            item.source_filename: item.output_filename
            for item in self.files
            if item.output_filename and item.output_filename != item.source_filename
        }

    def count(self, *statuses) -> int:
        return sum(item.status in statuses for item in self.files)

    @property
    def changed(self) -> int:
        return self.count(
            "normalized",
            "boost_limited",
            "peak_limited",
            "convert_only",
        )

    @property
    def normalized(self) -> int:
        return sum(
            bool(item.decision and abs(item.decision.gain_db) > 0.001)
            for item in self.files
            if item.output_filename
        )

    @property
    def converted(self) -> int:
        return sum(item.format_converted for item in self.files)

    @property
    def reused(self) -> int:
        return self.count("reused")

    @property
    def warnings(self) -> int:
        return self.count("missing", "unmeasurable")

    @property
    def errors(self) -> int:
        return self.count("error")

    @property
    def effectiveness(
        self,
    ) -> Optional[Tuple[LoudnessStatistics, LoudnessStatistics]]:
        """Actual before/after statistics for files measured during this run."""

        if self.target_lufs is None:
            return None
        pairs = [
            (item.measurement.integrated_lufs, item.output_measurement.integrated_lufs)
            for item in self.files
            if item.measurement is not None
            and item.output_measurement is not None
            and item.measurement.integrated_lufs is not None
            and item.output_measurement.integrated_lufs is not None
        ]
        if not pairs:
            return None
        before, after = zip(*pairs)
        before_stats = loudness_statistics(before, self.target_lufs)
        after_stats = loudness_statistics(after, self.target_lufs)
        if before_stats is None or after_stats is None:
            return None
        return before_stats, after_stats


class AudioNormalizationService:
    def __init__(
        self,
        backend: FFmpegBackend,
        media_directory: str,
        recipe: ProcessingRecipe,
        state=None,
    ):
        recipe.validate()
        self.backend = backend
        self.media_directory = media_directory
        self.recipe = recipe
        self.state = state

    def _verify(
        self,
        path: str,
        expected_lufs: Optional[float] = None,
    ) -> AudioMeasurement:
        verified = self.backend.analyze(path, self.recipe)
        if not verified.is_usable:
            raise FFmpegError("The normalized output could not be measured.")
        verification_ceiling = min(-0.5, self.recipe.true_peak_dbtp + 1.0)
        if verified.true_peak_dbtp > verification_ceiling:
            raise FFmpegError(
                "Output verification found insufficient peak headroom."
            )
        if (
            expected_lufs is not None
            and abs(verified.integrated_lufs - expected_lufs) > 1.5
        ):
            raise FFmpegError(
                "Output verification found an unexpected loudness result."
            )
        return verified

    def _infer_origin(self, filename: str) -> Optional[str]:
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        base = strip_processing_marker(stem)
        prefix = base + "."
        candidates = [
            candidate
            for candidate in os.listdir(self.media_directory)
            if candidate.startswith(prefix)
            and not marker_info(
                candidate.rsplit(".", 1)[0] if "." in candidate else candidate
            )
            and self._safe_origin(candidate)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _safe_origin(self, filename: Optional[str]) -> Optional[str]:
        if (
            not filename
            or filename != os.path.basename(filename)
            or filename in (".", "..")
            or marker_info(
                filename.rsplit(".", 1)[0] if "." in filename else filename
            )
        ):
            return None
        path = os.path.join(self.media_directory, filename)
        try:
            if (
                not os.path.isfile(path)
                or os.path.islink(path)
                or os.path.getsize(path) > 250 * 1024 * 1024
            ):
                return None
        except OSError:
            return None
        return filename

    def _remember_output(self, output_filename: str, origin: Optional[str]) -> None:
        if self.state is None:
            return
        output_path = os.path.join(self.media_directory, output_filename)
        self.state.remember(
            output_filename,
            sha256_file(output_path),
            self.recipe.token,
            "verified",
            origin=origin,
            source_path=output_path,
        )

    def process(
        self,
        filenames: Iterable[str],
        progress_callback: Optional[
            Callable[[int, int, str, Optional[FileResult]], None]
        ] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> BatchResult:
        result = BatchResult(target_lufs=self.recipe.target_lufs)
        ordered_filenames = sorted(set(filenames))
        total = len(ordered_filenames)
        for completed, filename in enumerate(ordered_filenames, start=1):
            if should_cancel is not None and should_cancel():
                result.cancelled = True
                break
            if progress_callback is not None:
                progress_callback(completed - 1, total, filename, None)
            item = self.process_one(filename)
            result.files.append(item)
            if progress_callback is not None:
                progress_callback(completed, total, filename, item)
        if self.state is not None:
            self.state.save()
        return result

    def process_one(self, filename: str) -> FileResult:
        if filename != os.path.basename(filename) or filename in ("", ".", ".."):
            return FileResult(
                source_filename=filename,
                status="error",
                detail="Unsafe media filename was rejected.",
            )
        source_path = os.path.join(self.media_directory, filename)
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if not os.path.isfile(source_path):
            cached = (
                self.state.status_for(
                    filename, "__missing__", self.recipe.token
                )
                if self.state is not None
                else None
            )
            if self.state is not None and not cached:
                self.state.remember(
                    filename, "__missing__", self.recipe.token, "missing"
                )
            return FileResult(
                source_filename=filename,
                status="missing_cached" if cached else "missing",
                detail=(
                    "Referenced media file is still missing."
                    if cached
                    else "Referenced media file was not found."
                ),
            )
        if os.path.islink(source_path):
            return FileResult(
                source_filename=filename,
                status="error",
                detail="Symbolic-link media is not processed.",
            )
        if os.path.getsize(source_path) > 250 * 1024 * 1024:
            return FileResult(
                source_filename=filename,
                status="error",
                detail="Media larger than 250 MB is not processed automatically.",
            )
        marker = marker_info(stem)
        if is_processed_for(stem, self.recipe):
            if self.state is not None:
                try:
                    origin = (
                        self._safe_origin(self.state.origin_for(filename))
                        or self._infer_origin(filename)
                    )
                    cached = self.state.unchanged_file(
                        filename, source_path, self.recipe.token
                    )
                    if cached:
                        source_hash, _cached_status = cached
                        if origin and not self.state.origin_for(filename):
                            self.state.remember(
                                filename,
                                source_hash,
                                self.recipe.token,
                                "verified",
                                origin=origin,
                                source_path=source_path,
                            )
                        return FileResult(
                            source_filename=filename,
                            status="already_processed",
                            detail="Already normalized with the current settings.",
                        )
                    source_hash = sha256_file(source_path)
                    if not self.state.status_for(
                        filename, source_hash, self.recipe.token
                    ):
                        self._verify(source_path)
                        self.state.remember(
                            filename,
                            source_hash,
                            self.recipe.token,
                            "verified",
                            origin=origin,
                            source_path=source_path,
                        )
                    else:
                        self.state.remember(
                            filename,
                            source_hash,
                            self.recipe.token,
                            "verified",
                            origin=origin,
                            source_path=source_path,
                        )
                except (FFmpegError, OSError, ValueError) as error:
                    return FileResult(
                        source_filename=filename,
                        status="error",
                        detail=f"Processed-file verification failed: {error}",
                    )
            return FileResult(
                source_filename=filename,
                status="already_processed",
                detail="Already normalized with the current settings.",
            )

        try:
            origin_filename = filename
            processing_filename = filename
            if marker:
                origin_filename = (
                    self._safe_origin(self.state.origin_for(filename))
                    if self.state is not None
                    else None
                ) or self._infer_origin(filename)
                if not origin_filename:
                    return FileResult(
                        source_filename=filename,
                        status="error",
                        detail=(
                            "This file was normalized with different settings, "
                            "but its original could not be identified. It was left unchanged."
                        ),
                    )
                processing_filename = origin_filename
                source_path = os.path.join(
                    self.media_directory, processing_filename
                )

            cached = (
                self.state.unchanged_file(
                    filename, source_path, self.recipe.token
                )
                if self.state is not None
                else None
            )
            if cached:
                _source_hash, cached_status = cached
                status = (
                    "already_ready_cached"
                    if cached_status in ("already_ready", "verified")
                    else (
                        "unmeasurable_cached"
                        if cached_status == "unmeasurable"
                        else cached_status
                    )
                )
                return FileResult(
                    source_filename=filename,
                    status=status,
                    detail=(
                        "Still too short or quiet to measure reliably."
                        if status == "unmeasurable_cached"
                        else "Unchanged and previously within the target tolerance."
                    ),
                )

            source_hash = sha256_file(source_path)
            if self.state is not None:
                cached_status = self.state.status_for(
                    filename, source_hash, self.recipe.token
                )
                if cached_status:
                    self.state.remember(
                        filename,
                        source_hash,
                        self.recipe.token,
                        cached_status,
                        source_path=source_path,
                    )
                    status = (
                        "already_ready_cached"
                        if cached_status in ("already_ready", "verified")
                        else (
                            "unmeasurable_cached"
                            if cached_status == "unmeasurable"
                            else cached_status
                        )
                    )
                    return FileResult(
                        source_filename=filename,
                        status=status,
                        detail=(
                            "Still too short or quiet to measure reliably."
                            if status == "unmeasurable_cached"
                            else "Unchanged and previously within the target tolerance."
                        ),
                    )
            source_extension = (
                processing_filename.rsplit(".", 1)[1].lower()
                if "." in processing_filename
                else ""
            )
            output_format = output_extension(
                processing_filename,
                self.recipe,
            )
            destination_name = output_filename(
                processing_filename,
                self.recipe,
                source_hash,
                concrete_format=output_format,
            )
            destination_path = os.path.join(self.media_directory, destination_name)
            if os.path.isfile(destination_path):
                # A deterministic name alone is not proof that an existing file
                # is the right output. Recompute the expected result before reuse
                # so a stale or unrelated but valid audio file is never linked.
                measurement = self.backend.analyze(source_path, self.recipe)
                needs_conversion = source_extension != output_format
                decision = decide_gain(measurement, self.recipe, needs_conversion)
                if not decision.process:
                    status = (
                        "unmeasurable"
                        if decision.outcome == "unmeasurable"
                        else "already_ready"
                    )
                    if self.state is not None:
                        self.state.remember(
                            filename,
                            source_hash,
                            self.recipe.token,
                            status,
                            source_path=source_path,
                        )
                    return FileResult(
                        source_filename=filename,
                        status=status,
                        detail=decision.detail,
                        measurement=measurement,
                        output_measurement=measurement,
                        decision=decision,
                    )
                output_measurement = self._verify(
                    destination_path, decision.expected_lufs
                )
                self._remember_output(destination_name, origin_filename)
                return FileResult(
                    source_filename=filename,
                    output_filename=destination_name,
                    status="reused",
                    detail="Reused an existing normalized copy.",
                    measurement=measurement,
                    output_measurement=output_measurement,
                )

            measurement = self.backend.analyze(source_path, self.recipe)
            needs_conversion = source_extension != output_format
            decision = decide_gain(measurement, self.recipe, needs_conversion)
            if not decision.process:
                status = (
                    "unmeasurable"
                    if decision.outcome == "unmeasurable"
                    else "already_ready"
                )
                item = FileResult(
                    source_filename=filename,
                    status=status,
                    detail=decision.detail,
                    measurement=measurement,
                    output_measurement=measurement,
                    decision=decision,
                )
                if self.state is not None:
                    self.state.remember(
                        filename,
                        source_hash,
                        self.recipe.token,
                        status,
                        source_path=source_path,
                    )
                return item

            self.backend.transform(
                source_path,
                destination_path,
                decision.gain_db,
                self.recipe,
                output_format=output_format,
            )
            try:
                output_measurement = self._verify(
                    destination_path, decision.expected_lufs
                )
            except (FFmpegError, OSError, ValueError):
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                raise
            self._remember_output(
                destination_name,
                origin_filename,
            )
            return FileResult(
                source_filename=filename,
                output_filename=destination_name,
                status=decision.outcome,
                detail=decision.detail,
                measurement=measurement,
                output_measurement=output_measurement,
                decision=decision,
                format_converted=needs_conversion,
            )
        except (FFmpegError, OSError, ValueError) as error:
            return FileResult(
                source_filename=filename,
                status="error",
                detail=str(error),
            )
