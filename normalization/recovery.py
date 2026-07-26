"""Plan safe note-reference restoration without deleting media."""

from dataclasses import dataclass, field
import os
from typing import Dict, Iterable

from .core import marker_info, strip_processing_marker


@dataclass
class RevertPlan:
    replacements: Dict[str, str] = field(default_factory=dict)
    already_original: int = 0
    unresolved: Dict[str, str] = field(default_factory=dict)


def _safe_originals(media_directory: str) -> set[str]:
    originals = set()
    try:
        candidates = os.listdir(media_directory)
    except OSError:
        return originals
    for filename in candidates:
        if (
            not filename
            or filename != os.path.basename(filename)
            or filename in (".", "..")
        ):
            continue
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if marker_info(stem):
            continue
        path = os.path.join(media_directory, filename)
        try:
            if os.path.isfile(path) and not os.path.islink(path):
                originals.add(filename)
        except OSError:
            continue
    return originals


def plan_revert(
    media_directory: str,
    filenames: Iterable[str],
    state,
) -> RevertPlan:
    """Resolve each generated filename to one unambiguous retained original."""
    plan = RevertPlan()
    originals = _safe_originals(media_directory)
    originals_by_stem: Dict[str, list[str]] = {}
    for original in originals:
        stem = original.rsplit(".", 1)[0] if "." in original else original
        originals_by_stem.setdefault(stem, []).append(original)

    for filename in sorted(set(filenames)):
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if not marker_info(stem):
            plan.already_original += 1
            continue

        recorded = state.origin_for(filename) if state is not None else None
        if recorded in originals:
            plan.replacements[filename] = recorded
            continue
        if recorded:
            plan.unresolved[filename] = (
                f"The recorded original is missing: {recorded}"
            )
            continue

        original_stem = strip_processing_marker(stem)
        inferred = originals_by_stem.get(original_stem, [])
        if len(inferred) == 1:
            plan.replacements[filename] = inferred[0]
        elif len(inferred) > 1:
            plan.unresolved[filename] = (
                "Multiple retained originals have the same base name."
            )
        else:
            plan.unresolved[filename] = (
                "No retained original could be identified."
            )
    return plan
