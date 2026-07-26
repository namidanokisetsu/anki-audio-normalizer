"""Pure helpers for composing a safe, understandable Anki search scope."""

import re
from typing import List, Tuple


_LEGACY_DECK_PATTERNS = (
    re.compile(r'^\s*deck:"((?:\\.|[^"])*)"(?:\s+|$)', re.IGNORECASE),
    re.compile(r'^\s*"deck:((?:\\.|[^"])*)"(?:\s+|$)', re.IGNORECASE),
    re.compile(r"^\s*deck:([^\s()]+)(?:\s+|$)", re.IGNORECASE),
)


def _unescape_search_value(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def deck_search(deck_name: str) -> str:
    name = str(deck_name or "").strip()
    if not name:
        return ""
    escaped = name.replace("\\", r"\\").replace('"', r"\"")
    return f'deck:"{escaped}"'


def effective_search(deck_name: str, additional_filters: str) -> str:
    deck_filter = deck_search(deck_name)
    additional = str(additional_filters or "").strip()
    if deck_filter and additional:
        return f"{deck_filter} ({additional})"
    return deck_filter or additional


def browser_scope_search(current_search: str, note_ids) -> str:
    """Intersect a Browser search with an exact, safely encoded note selection."""

    search = str(current_search or "").replace("\n", " ").strip()
    selected = []
    for note_id in note_ids or ():
        try:
            number = int(note_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Browser selection contained an invalid note ID.") from error
        if number <= 0:
            raise ValueError("Browser selection contained an invalid note ID.")
        selected.append(number)
    selected = sorted(set(selected))
    if not selected:
        return search
    note_filter = "nid:" + ",".join(str(note_id) for note_id in selected)
    return f"({search}) ({note_filter})" if search else note_filter


def split_legacy_scope(deck_name: str, query: str) -> Tuple[str, str]:
    """Move a leading legacy deck filter into the first-class deck setting."""

    deck = str(deck_name or "").strip()
    additional = str(query or "").strip()
    if deck or not additional:
        return deck, additional
    for pattern in _LEGACY_DECK_PATTERNS:
        match = pattern.match(additional)
        if match:
            return (
                _unescape_search_value(match.group(1)),
                additional[match.end():].strip(),
            )
    return "", additional


def available_deck_names(deck_manager) -> List[str]:
    """Return deck names across current and older Anki deck-manager APIs."""

    try:
        values = deck_manager.all_names_and_ids()
        names = [
            str(item.name() if callable(item.name) else item.name)
            for item in values
        ]
    except (AttributeError, TypeError):
        try:
            names = [str(item["name"]) for item in deck_manager.all()]
        except (AttributeError, KeyError, TypeError):
            names = []
    return sorted(set(names), key=str.casefold)
