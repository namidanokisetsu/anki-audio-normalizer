"""Anki-facing media discovery and reference updates."""

from collections import defaultdict
from dataclasses import dataclass
import html
import os
import re
from typing import Dict, Iterable, List


SOUND_TAG_RE = re.compile(r"\[sound:([^\]\r\n]+)\]", re.IGNORECASE)
FILENAME_MARKER_RE = re.compile(
    r"(<span\b(?=[^>]*\bdata-audio-normalizer-filename\b)[^>]*>)"
    r"([^<]*)"
    r"(</span>)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MediaReference:
    note_id: int
    field_name: str
    filename: str


@dataclass
class DiscoveryResult:
    references: List[MediaReference]
    unsafe_references: int = 0

    @property
    def filenames(self):
        return sorted({reference.filename for reference in self.references})


def _safe_media_filename(filename: str) -> bool:
    return (
        bool(filename)
        and filename == os.path.basename(filename)
        and filename not in (".", "..")
        and "\x00" not in filename
    )


def discover_media_references(
    collection, note_ids: Iterable[int], progress_callback=None
) -> DiscoveryResult:
    references = []
    unsafe = 0
    seen = set()
    for index, note_id in enumerate(note_ids):
        if progress_callback is not None and index and index % 250 == 0:
            progress_callback()
        note = collection.get_note(note_id)
        if note is None:
            continue
        for field_name in note.keys():
            for filename in SOUND_TAG_RE.findall(note[field_name]):
                filename = filename.strip()
                if not _safe_media_filename(filename):
                    unsafe += 1
                    continue
                key = (note_id, field_name, filename)
                if key not in seen:
                    seen.add(key)
                    references.append(MediaReference(*key))
    return DiscoveryResult(references=references, unsafe_references=unsafe)


def replace_media_references(
    collection,
    references,
    replacements: Dict[str, str],
    undo_label: str = "Normalize card audio",
) -> int:
    """Update sound tags and any opt-in filename labels on the same note."""

    grouped = defaultdict(list)
    for reference in references:
        if reference.filename in replacements:
            grouped[reference.note_id].append(reference)

    updated_notes = 0
    undo_entry = None
    if grouped and hasattr(collection, "add_custom_undo_entry"):
        try:
            undo_entry = collection.add_custom_undo_entry(undo_label)
        except Exception:
            undo_entry = None

    for note_id, note_references in grouped.items():
        note = collection.get_note(note_id)
        if note is None:
            continue
        changed = False
        applied_replacements = set()
        for reference in note_references:
            if reference.field_name not in note.keys():
                continue
            old_tag = f"[sound:{reference.filename}]"
            new_tag = f"[sound:{replacements[reference.filename]}]"
            current = note[reference.field_name]
            replaced = re.sub(
                re.escape(old_tag),
                lambda _match: new_tag,
                current,
                flags=re.IGNORECASE,
            )
            if replaced != current:
                note[reference.field_name] = replaced
                changed = True
                applied_replacements.add(reference.filename)
        for field_name in note.keys():
            current = note[field_name]

            def replace_filename_label(match):
                visible_filename = html.unescape(match.group(2))
                if visible_filename not in applied_replacements:
                    return match.group(0)
                replacement = replacements[visible_filename]
                return match.group(1) + html.escape(replacement) + match.group(3)

            replaced = FILENAME_MARKER_RE.sub(replace_filename_label, current)
            if replaced != current:
                note[field_name] = replaced
                changed = True
        if changed:
            collection.update_note(note)
            updated_notes += 1

    if undo_entry is not None and hasattr(collection, "merge_undo_entries"):
        try:
            collection.merge_undo_entries(undo_entry)
        except Exception:
            pass
    return updated_notes
