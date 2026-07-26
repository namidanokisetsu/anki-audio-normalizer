"""Pure audio-normalization rules.

This module deliberately has no Anki, Qt, filesystem, or subprocess imports.
It defines the processing recipe and bounded gain decision.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import statistics
from typing import Optional


PROCESSOR_VERSION = 1
MARKER_PREFIX = "__anorm"
_MARKER_RE = re.compile(
    rf"(?P<prefix>{re.escape(MARKER_PREFIX)}|__vva)_v(?P<version>\d+)_"
    r"(?P<recipe>[0-9a-f]{10})_(?P<source>[0-9a-f]{12})$"
)


@dataclass(frozen=True)
class ProcessingRecipe:
    """A conservative, transparent audio-processing recipe."""

    target_lufs: float = -18.0
    true_peak_dbtp: float = -2.0
    # None permits any boost required by the target and true-peak ceiling.
    max_boost_db: Optional[float] = None
    tolerance_lu: float = 0.5
    # "source" preserves the source container/extension when gain is changed.
    output_format: str = "mp3"
    normalize: bool = True

    def validate(self) -> None:
        if not -30.0 <= self.target_lufs <= -16.0:
            raise ValueError("Target loudness must be between -30 and -16 LUFS.")
        if not -9.0 <= self.true_peak_dbtp <= -1.0:
            raise ValueError("True-peak ceiling must be between -9 and -1 dBTP.")
        if self.max_boost_db is not None and not 0.0 <= self.max_boost_db <= 18.0:
            raise ValueError("Maximum boost must be between 0 and 18 dB.")
        if not 0.0 <= self.tolerance_lu <= 2.0:
            raise ValueError("Tolerance must be between 0 and 2 LU.")
        if self.output_format not in ("mp3", "source"):
            raise ValueError("Output must be MP3 or keep the source extension.")

    @property
    def token(self) -> str:
        self.validate()
        payload = {
            "processor_version": PROCESSOR_VERSION,
            **asdict(self),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:10]


@dataclass(frozen=True)
class AudioMeasurement:
    integrated_lufs: Optional[float]
    true_peak_dbtp: Optional[float]

    @property
    def is_usable(self) -> bool:
        return (
            self.integrated_lufs is not None
            and self.true_peak_dbtp is not None
            and math.isfinite(self.integrated_lufs)
            and math.isfinite(self.true_peak_dbtp)
        )


@dataclass(frozen=True)
class LoudnessStatistics:
    count: int
    mean_lufs: float
    standard_deviation_lu: float
    mean_distance_from_target_lu: float


def loudness_statistics(values, target_lufs: float) -> Optional[LoudnessStatistics]:
    """Summarize finite loudness readings using population dispersion."""

    usable = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not usable:
        return None
    return LoudnessStatistics(
        count=len(usable),
        mean_lufs=statistics.fmean(usable),
        standard_deviation_lu=statistics.pstdev(usable),
        mean_distance_from_target_lu=statistics.fmean(
            abs(value - target_lufs) for value in usable
        ),
    )


@dataclass(frozen=True)
class GainDecision:
    process: bool
    gain_db: float
    outcome: str
    detail: str
    expected_lufs: Optional[float] = None
    expected_true_peak_dbtp: Optional[float] = None


def decide_gain(
    measurement: AudioMeasurement,
    recipe: ProcessingRecipe,
    needs_conversion: bool,
) -> GainDecision:
    """Choose a single, bounded gain adjustment.

    No compressor, dynamic limiter, or denoiser is involved. The peak ceiling
    is enforced by reducing the permitted constant gain.
    """

    recipe.validate()
    if not recipe.normalize:
        return GainDecision(
            process=needs_conversion,
            gain_db=0.0,
            outcome="convert_only" if needs_conversion else "already_ready",
            detail="Format conversion only.",
        )
    if not measurement.is_usable:
        return GainDecision(
            process=False,
            gain_db=0.0,
            outcome="unmeasurable",
            detail="The clip is too short or has no measurable programme loudness.",
        )

    integrated = float(measurement.integrated_lufs)
    true_peak = float(measurement.true_peak_dbtp)
    wanted_gain = recipe.target_lufs - integrated

    if abs(wanted_gain) <= recipe.tolerance_lu:
        gain = 0.0
        outcome = "convert_only" if needs_conversion else "already_ready"
        detail = "Already within the target loudness tolerance."
    else:
        gain = wanted_gain
        outcome = "normalized"
        detail = "Adjusted to the requested loudness."

        if recipe.max_boost_db is not None and gain > recipe.max_boost_db:
            gain = recipe.max_boost_db
            outcome = "boost_limited"
            detail = "Boost was limited by the safety cap."

    peak_safe_gain = recipe.true_peak_dbtp - true_peak
    if gain > peak_safe_gain:
        gain = peak_safe_gain
        outcome = "peak_limited"
        detail = "Gain was limited to preserve true-peak headroom."

    # Rounding makes filenames/reports stable without meaningfully changing gain.
    gain = round(gain, 3)
    process = needs_conversion or abs(gain) > 0.001
    return GainDecision(
        process=process,
        gain_db=gain,
        outcome=outcome,
        detail=detail,
        expected_lufs=round(integrated + gain, 2),
        expected_true_peak_dbtp=round(true_peak + gain, 2),
    )


def marker_for(recipe: ProcessingRecipe, source_hash: str) -> str:
    return (
        f"{MARKER_PREFIX}_v{PROCESSOR_VERSION}_"
        f"{recipe.token}_{source_hash[:12].lower()}"
    )


def marker_info(filename_stem: str):
    match = _MARKER_RE.search(filename_stem)
    return match.groupdict() if match else None


def is_processed_for(filename_stem: str, recipe: ProcessingRecipe) -> bool:
    info = marker_info(filename_stem)
    return bool(
        info
        and int(info["version"]) == PROCESSOR_VERSION
        and info["recipe"] == recipe.token
    )


def strip_processing_marker(filename_stem: str) -> str:
    return _MARKER_RE.sub("", filename_stem)


def output_extension(
    original_filename: str,
    recipe: ProcessingRecipe,
    preferred: Optional[str] = None,
) -> str:
    source_extension = (
        original_filename.rsplit(".", 1)[1].lower()
        if "." in original_filename
        else ""
    )
    extension = (preferred or "").lower().lstrip(".") or (
        source_extension if recipe.output_format == "source" else recipe.output_format
    )
    if not extension:
        raise ValueError("The source file has no format to preserve.")
    return extension


def output_filename(
    original_filename: str,
    recipe: ProcessingRecipe,
    source_hash: str,
    concrete_format: Optional[str] = None,
) -> str:
    """Return a deterministic, cross-platform media filename."""

    stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    extension = output_extension(
        original_filename, recipe, preferred=concrete_format
    )
    stem = strip_processing_marker(stem)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip().strip(".")
    marker_and_extension = f"{marker_for(recipe, source_hash)}.{extension}"
    # Linux limits a filename component to 255 bytes, while Windows applies a
    # similarly sized character limit. Keep some margin and truncate on a UTF-8
    # boundary so long non-ASCII media names remain portable.
    stem_budget = max(1, 240 - len(marker_and_extension.encode("utf-8")))
    stem = (stem or "audio").encode("utf-8")[:stem_budget].decode(
        "utf-8", errors="ignore"
    )
    return f"{stem or 'audio'}{marker_and_extension}"
