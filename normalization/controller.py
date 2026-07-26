"""Automatic loudness normalization for matching Anki card audio."""

from __future__ import annotations

import datetime
import hashlib
import html
import os
import tempfile
import threading
import time

from aqt import gui_hooks, mw, qt
from aqt.operations import CollectionOp, QueryOp
from aqt.utils import tooltip
from anki.collection import OpChanges

from .. import ffmpeg
from .core import (
    ProcessingRecipe,
    decide_gain,
    is_processed_for,
    loudness_statistics,
    output_extension,
)
from .ffmpeg_backend import FFmpegBackend, FFmpegError
from .media import discover_media_references, replace_media_references
from .recovery import plan_revert
from .scope import (
    available_deck_names,
    browser_scope_search,
    effective_search,
    split_legacy_scope,
)
from .service import AudioNormalizationService, BatchResult
from .state import ProcessingState


DEFAULT_SETTINGS = {
    "enabled": False,
    "deck": "",
    "query": "",
    "output_format": "mp3",
    "target_lufs": -18.0,
    "max_boost_db": None,
    "startup": True,
    "after_sync": True,
    "after_changes": True,
    "automatic_display": "compact",
}

ADDON_ROOT = os.path.dirname(os.path.dirname(__file__))
# Preview analysis decodes each complete clip and does not populate the run cache.
PREVIEW_SAMPLE_LIMIT = 30
PREVIEW_FILE_TIMEOUT_SECONDS = 15
PREVIEW_TIME_BUDGET_SECONDS = 45

_running = False
_scheduled = False
_scheduled_allow_disabled = False
_pending = False
_media_sync_running = False
_post_sync_waiting = False
_updating_note_references = False
_schedule_generation = 0
_active_cancel_event = None
_pending_settings_override = None
_OPERATION_INITIATOR = object()


def _profile_name():
    profile_manager = getattr(mw, "pm", None)
    value = getattr(profile_manager, "name", "") if profile_manager else ""
    return str(value() if callable(value) else value)


def _addon_module():
    return (__package__ or __name__).split(".")[0]


def _addon_config():
    return mw.addonManager.getConfig(_addon_module()) or {}


def load_settings():
    config = _addon_config()
    profiles = config.get("profiles", {}) if isinstance(config, dict) else {}
    if not isinstance(profiles, dict):
        profiles = {}
    stored = profiles.get(_profile_name(), {})
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(stored, dict):
        settings.update({key: stored[key] for key in settings if key in stored})
    for key in ("enabled", "startup", "after_sync", "after_changes"):
        if not isinstance(settings.get(key), bool):
            settings[key] = DEFAULT_SETTINGS[key]
    for key in ("deck", "query"):
        if not isinstance(settings.get(key), str):
            settings[key] = DEFAULT_SETTINGS[key]
    if settings.get("output_format") not in ("mp3", "source"):
        settings["output_format"] = DEFAULT_SETTINGS["output_format"]
    for key, minimum, maximum in (("target_lufs", -30.0, -16.0),):
        try:
            value = float(settings[key])
        except (TypeError, ValueError):
            value = DEFAULT_SETTINGS[key]
        settings[key] = (
            value if minimum <= value <= maximum else DEFAULT_SETTINGS[key]
        )
    boost = settings.get("max_boost_db")
    if boost is not None:
        try:
            boost = float(boost)
        except (TypeError, ValueError):
            boost = DEFAULT_SETTINGS["max_boost_db"]
        if not 0.0 <= boost <= 18.0:
            boost = DEFAULT_SETTINGS["max_boost_db"]
    settings["max_boost_db"] = boost
    settings["deck"], settings["query"] = split_legacy_scope(
        settings.get("deck", ""), settings.get("query", "")
    )
    if settings.get("automatic_display") not in ("compact", "window"):
        settings["automatic_display"] = "compact"
    return settings


def save_settings(settings):
    config = _addon_config()
    if not isinstance(config, dict):
        config = {}
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[_profile_name()] = {
        key: settings.get(key, default) for key, default in DEFAULT_SETTINGS.items()
    }
    config["version"] = 1
    config["profiles"] = profiles
    mw.addonManager.writeConfig(_addon_module(), config)


def recipe_from_settings(settings=None):
    settings = settings or load_settings()
    boost = settings.get("max_boost_db")
    return ProcessingRecipe(
        target_lufs=float(settings.get("target_lufs", -18.0)),
        true_peak_dbtp=-2.0,
        max_boost_db=None if boost is None else float(boost),
        output_format=str(settings.get("output_format", "mp3")),
        normalize=True,
    )


def is_enabled():
    try:
        return bool(load_settings()["enabled"])
    except Exception:
        return False


def _state_path():
    profile_hash = hashlib.sha256(_profile_name().encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        ADDON_ROOT,
        "user_files",
        f"audio_normalizer_{profile_hash}.json",
    )


def _report_path():
    profile_hash = hashlib.sha256(_profile_name().encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        ADDON_ROOT,
        "user_files",
        f"audio_normalizer_report_{profile_hash}.txt",
    )


def _last_report_summary():
    try:
        with open(_report_path(), "r", encoding="utf-8") as report:
            return report.readline().strip()
    except OSError:
        return "No processing run has completed yet."


def _save_report(lines):
    path = _report_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".audio_report_", suffix=".txt", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as report:
            report.write("\n".join(lines) + "\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _effectiveness_lines(effectiveness, heading):
    if effectiveness is None:
        return []
    before, after = effectiveness
    return [
        f"{heading} ({before.count} files):",
        (
            f"Before: mean {before.mean_lufs:.1f} LUFS, "
            f"standard deviation {before.standard_deviation_lu:.1f} LU"
        ),
        (
            f"After: mean {after.mean_lufs:.1f} LUFS, "
            f"standard deviation {after.standard_deviation_lu:.1f} LU"
        ),
        (
            "Mean distance from target: "
            f"{before.mean_distance_from_target_lu:.1f} → "
            f"{after.mean_distance_from_target_lu:.1f} LU"
        ),
    ]


def _write_report(batch, updated_notes, reason):
    run_state = (
        f"Stopped after {len(batch.files)} files"
        if batch.cancelled
        else "Completed"
    )
    lines = [
        (
            f"Last run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} — "
            f"{run_state}; "
            f"{batch.changed} new files: {batch.normalized} gain-adjusted, "
            f"{batch.converted} format-converted; {batch.reused} reused, "
            f"{updated_notes} notes updated, {batch.warnings} warnings, "
            f"{batch.errors} errors."
        ),
        f"Trigger: {reason}",
        f"FFmpeg: {getattr(ffmpeg.ffmpeg_provider, 'version', None) or 'unknown'}",
        "",
    ]
    effectiveness = _effectiveness_lines(
        batch.effectiveness,
        "Measured loudness effectiveness",
    )
    if effectiveness:
        lines += effectiveness + [""]
    for item in batch.files:
        destination = f" -> {item.output_filename}" if item.output_filename else ""
        lines.append(
            f"{item.status}: {item.source_filename}{destination} — {item.detail}"
        )
    if not batch.files:
        lines.append("Everything was already up to date.")
    _save_report(lines)


def _write_revert_report(plan, updated_notes, scope):
    lines = [
        (
            f"Last action: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} — "
            f"Restored {len(plan.replacements)} unique audio files from originals; "
            f"{updated_notes} notes updated, {len(plan.unresolved)} unresolved."
        ),
        f"Scope: {scope or 'entire collection'}",
        "",
    ]
    for generated, original in sorted(plan.replacements.items()):
        lines.append(f"restored: {generated} -> {original}")
    for generated, detail in sorted(plan.unresolved.items()):
        lines.append(f"unresolved: {generated} — {detail}")
    _save_report(lines)


def _ensure_backend(timeout_seconds=180):
    executable = ffmpeg.ffmpeg_provider.discover()
    if executable is None:
        executable = ffmpeg.ffmpeg_provider.ensure_available()
    if executable is None:
        detail = getattr(ffmpeg.ffmpeg_provider, "last_error", None)
        raise FFmpegError(
            "FFmpeg was not found or lacks a required audio filter or encoder. "
            "Install or update FFmpeg, restart Anki, and try again. See the "
            "add-on README for platform-specific instructions."
            + (f" Details: {detail}" if detail else "")
        )
    return FFmpegBackend(executable, timeout_seconds=timeout_seconds)


def _matching_note_ids(settings, collection=None):
    collection = collection or mw.col
    return list(collection.find_notes(search_from_settings(settings)))


def search_from_settings(settings):
    return effective_search(
        str(settings.get("deck", "")),
        str(settings.get("query", "")),
    )


def _discover(settings, collection=None, progress_callback=None):
    collection = collection or mw.col
    note_ids = _matching_note_ids(settings, collection)
    return note_ids, discover_media_references(
        collection,
        note_ids,
        progress_callback=progress_callback,
    )


def _preview_scope(settings, collection, progress_callback=None):
    note_ids, discovery = _discover(
        settings,
        collection=collection,
        progress_callback=progress_callback,
    )
    return len(note_ids), discovery, collection.media.dir()


def _preview_text(
    settings,
    preview_scope,
    progress_callback=None,
):
    note_count, discovery, media_directory = preview_scope
    filenames = discovery.filenames
    recipe = recipe_from_settings(settings)

    if len(filenames) <= PREVIEW_SAMPLE_LIMIT:
        sample = filenames
    else:
        sample = [
            filenames[
                round(
                    index
                    * (len(filenames) - 1)
                    / (PREVIEW_SAMPLE_LIMIT - 1)
                )
            ]
            for index in range(PREVIEW_SAMPLE_LIMIT)
        ]
    if progress_callback is not None:
        progress_callback(0, len(sample))
    backend = _ensure_backend(timeout_seconds=PREVIEW_FILE_TIMEOUT_SECONDS)
    deadline = time.monotonic() + PREVIEW_TIME_BUDGET_SECONDS
    measured_before = []
    projected_after = []
    would_adjust = 0
    unmeasurable = 0
    missing = 0
    rejected = 0
    already_processed = 0
    checked = 0
    for index, filename in enumerate(sample, start=1):
        if time.monotonic() >= deadline:
            break
        checked += 1
        if progress_callback is not None:
            progress_callback(index, len(sample))
        path = os.path.join(media_directory, filename)
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        if not os.path.isfile(path):
            missing += 1
            continue
        try:
            if os.path.islink(path) or os.path.getsize(path) > 250 * 1024 * 1024:
                rejected += 1
                continue
        except OSError:
            rejected += 1
            continue
        if is_processed_for(stem, recipe):
            already_processed += 1
            continue
        try:
            measurement = backend.analyze(path, recipe)
            source_extension = (
                filename.rsplit(".", 1)[1].lower() if "." in filename else ""
            )
            decision = decide_gain(
                measurement,
                recipe,
                source_extension != output_extension(filename, recipe),
            )
            if (
                measurement.integrated_lufs is not None
                and decision.expected_lufs is not None
            ):
                measured_before.append(measurement.integrated_lufs)
                projected_after.append(decision.expected_lufs)
            if decision.process:
                would_adjust += 1
            elif decision.outcome == "unmeasurable":
                unmeasurable += 1
        except (FFmpegError, OSError, ValueError):
            unmeasurable += 1

    lines = [
        (
            f"Scope: {search_from_settings(settings) or 'entire collection'}"
        ),
        f"Matching notes: {note_count}",
        f"Unique referenced audio files: {len(filenames)}",
        f"FFmpeg: {getattr(ffmpeg.ffmpeg_provider, 'version', None) or 'unknown'}",
        "",
        f"Audio files checked now: {checked}",
        f"Would normalize or convert: {would_adjust}",
        f"Already normalized: {already_processed}",
        f"Missing, unsafe, or unmeasurable: {missing + rejected + unmeasurable}",
    ]
    before_stats = loudness_statistics(measured_before, recipe.target_lufs)
    after_stats = loudness_statistics(projected_after, recipe.target_lufs)
    effectiveness = (
        (before_stats, after_stats)
        if before_stats is not None and after_stats is not None
        else None
    )
    if effectiveness:
        lines += [
            "",
            * _effectiveness_lines(
                effectiveness,
                "Projected loudness effectiveness for the measurable sample",
            ),
            (
                f"Before range: {min(measured_before):.1f} to "
                f"{max(measured_before):.1f} LUFS"
            ),
            f"Target: {recipe.target_lufs:.0f} LUFS",
        ]
    if len(filenames) > len(sample):
        lines += [
            "",
            f"For a quick check, this inspected {PREVIEW_SAMPLE_LIMIT} files "
            "spread across the list. "
            "Save and normalize checks every matching file.",
        ]
    if checked < len(sample):
        lines += [
            "",
            (
                "Preview time limit reached; "
                f"{len(sample) - checked} sampled files were not checked."
            ),
        ]
    if discovery.unsafe_references:
        lines += [
            "",
            f"Unsafe media references ignored: {discovery.unsafe_references}",
        ]
    return "\n".join(lines)


class NormalizationStatusDialog(qt.QDialog):
    _QUIET_STATUSES = {
        "already_processed",
        "already_ready",
        "already_ready_cached",
        "missing_cached",
        "unmeasurable_cached",
    }

    def __init__(self, parent=None):
        super().__init__(parent or mw)
        self._cancel_callback = None
        self._show_automatic_details = False
        self.setWindowTitle("Card Audio Normalizer")
        self.setModal(False)
        self.setMinimumSize(520, 320)
        self.resize(760, 500)
        self.setSizeGripEnabled(True)

        layout = qt.QVBoxLayout(self)
        self.summary = qt.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.progress = qt.QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.log = qt.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Run details will appear here.")
        self.log.document().setMaximumBlockCount(10000)
        layout.addWidget(self.log, 1)

        buttons = qt.QDialogButtonBox()
        self.stop_button = buttons.addButton(
            "Stop after current file",
            qt.QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.stop_button.clicked.connect(self._request_stop)
        buttons.addButton(qt.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        layout.addWidget(buttons)

    def _show(self, activate=False):
        self.show()
        if activate:
            self.raise_()
            self.activateWindow()

    def _reason_label(self, reason):
        labels = {
            "manual": "Manual run",
            "startup": "Automatic startup run",
            "after_sync": "Automatic post-sync run",
            "note_change": "Automatic run after note changes",
            "pending": "Automatic queued run",
        }
        return labels.get(reason, f"Automatic run ({reason})")

    def begin_run(self, reason, cancel_callback, automatic_display="compact"):
        reason_label = self._reason_label(reason)
        self._cancel_callback = cancel_callback
        self._show_automatic_details = automatic_display == "window"
        self.stop_button.setEnabled(True)
        self.setWindowTitle("Card Audio Normalizer — running")
        self.summary.setText(
            f"{reason_label}: preparing audio tools and finding matching audio…"
        )
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("")
        self.log.clear()
        self.log.appendPlainText(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
            f"{reason_label} started"
        )
        if reason == "manual":
            self._show(activate=True)

    def set_total(self, total, reason):
        reason_label = self._reason_label(reason)
        self.summary.setText(
            f"{reason_label}: checking {total:,} unique audio files."
        )
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setFormat(f"0 / {total:,}")
        self.log.appendPlainText(f"Files to check: {total:,}\n")

    def update_batch(self, latest, finished_items):
        if latest is None:
            return
        completed, total, filename, item = latest
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        self.progress.setFormat(f"{completed:,} / {total:,}")
        if item is None:
            self.summary.setText(
                f"Processing file {completed + 1:,} of {total:,}: {filename}"
            )
        else:
            self.summary.setText(
                f"Processed {completed:,} of {total:,}: {filename}"
            )

        should_show = False
        log_lines = []
        for item_completed, item_total, _item_filename, finished in finished_items:
            destination = (
                f" → {finished.output_filename}"
                if finished.output_filename
                else ""
            )
            status = finished.status.replace("_", " ")
            log_lines.append(
                f"[{item_completed:,}/{item_total:,}] {status}: "
                f"{finished.source_filename}{destination}\n"
                f"    {finished.detail}"
            )
            if finished.status not in self._QUIET_STATUSES:
                should_show = True
        if log_lines:
            self.log.appendPlainText("\n".join(log_lines))
        if (
            should_show
            and self._show_automatic_details
            and not self.isVisible()
        ):
            self._show(activate=False)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def finish_run(self, batch, updated_notes, total):
        completed = len(batch.files)
        self._cancel_callback = None
        self.stop_button.setEnabled(False)
        self.setWindowTitle(
            "Card Audio Normalizer — stopped"
            if batch.cancelled
            else "Card Audio Normalizer — completed"
        )
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(completed)
            self.progress.setFormat(f"{completed:,} / {total:,}")
        else:
            self.progress.setVisible(False)
        state = "Stopped" if batch.cancelled else "Completed"
        summary = (
            f"{state}: {batch.changed:,} new files "
            f"({batch.normalized:,} gain-adjusted, "
            f"{batch.converted:,} format-converted), "
            f"{batch.reused:,} reused, {updated_notes:,} notes updated, "
            f"{batch.warnings:,} warnings, {batch.errors:,} errors."
        )
        effectiveness = _effectiveness_lines(
            batch.effectiveness,
            "Measured loudness effectiveness",
        )
        if effectiveness:
            summary += "\n" + "\n".join(effectiveness)
        self.summary.setText(summary)
        self.log.appendPlainText(
            "\n"
            + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + "  "
            + summary
        )

    def _request_stop(self):
        if self._cancel_callback is None:
            return
        self._cancel_callback()
        self.stop_button.setEnabled(False)
        self.summary.setText(
            "Stopping after the current audio file finishes…"
        )

    def show_error(self, message, activate=True, append=True, show=True):
        self._cancel_callback = None
        self.stop_button.setEnabled(False)
        self.setWindowTitle("Card Audio Normalizer — error")
        self.summary.setText("Normalization could not complete.")
        self.progress.setVisible(False)
        if not append:
            self.log.clear()
        self.log.appendPlainText(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
            f"ERROR: {message}"
        )
        if show:
            self._show(activate=activate)

    def show_report(self, text):
        self._cancel_callback = None
        self.stop_button.setEnabled(False)
        self.setWindowTitle("Card Audio Normalizer — last run")
        self.summary.setText("Saved report from the most recently completed run.")
        self.progress.setVisible(False)
        self.log.setPlainText(text)
        self._show(activate=True)


class ProgressRelay:
    """Coalesce worker updates so large cached runs do not flood Qt."""

    def __init__(self, dialog):
        self.dialog = dialog
        self.lock = threading.Lock()
        self.latest = None
        self.finished_items = []
        self.timer_armed = False
        self.finished = False

    def push(self, completed, total, filename, item):
        should_arm = False
        with self.lock:
            if self.finished:
                return
            self.latest = (completed, total, filename, item)
            if item is not None:
                self.finished_items.append(
                    (completed, total, filename, item)
                )
            if not self.timer_armed:
                self.timer_armed = True
                should_arm = True
        if should_arm:
            if hasattr(mw, "taskman"):
                mw.taskman.run_on_main(self._arm_timer)
            else:
                self._arm_timer()

    def _arm_timer(self):
        with self.lock:
            if self.finished:
                return
        qt.QTimer.singleShot(100, self.flush)

    def _take_updates(self):
        with self.lock:
            latest = self.latest
            finished_items = self.finished_items
            self.finished_items = []
            self.timer_armed = False
        return latest, finished_items

    def flush(self):
        with self.lock:
            if self.finished:
                return
        latest, finished_items = self._take_updates()
        self.dialog.update_batch(latest, finished_items)

    def finish(self):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            latest = self.latest
            finished_items = self.finished_items
            self.finished_items = []
        self.dialog.update_batch(latest, finished_items)


def _status_dialog():
    dialog = getattr(mw, "_audio_normalizer_status_dialog", None)
    if dialog is None:
        dialog = NormalizationStatusDialog(mw)
        mw._audio_normalizer_status_dialog = dialog
    return dialog


class CardAudioNormalizerDialog(qt.QDialog):
    def __init__(self, parent=None, external_scope=None):
        super().__init__(parent or mw)
        self.original = load_settings()
        self.external_scope = None
        self.setWindowTitle("Normalize Card Audio")
        self.resize(680, 520)

        layout = qt.QVBoxLayout()
        title = qt.QLabel("<h2>Normalize card audio loudness</h2>")
        layout.addWidget(title)
        description = qt.QLabel(
            "Reduce loudness jumps between referenced audio clips. Noise, timing, "
            "and speech remain intact, and original media files are kept."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.scope_notice = qt.QLabel()
        self.scope_notice.setWordWrap(True)
        self.scope_notice.setStyleSheet(
            "padding: 10px; border: 1px solid palette(mid); border-radius: 5px;"
        )
        self.scope_notice.setVisible(False)
        layout.addWidget(self.scope_notice)

        manual_group = qt.QGroupBox("Audio to normalize")
        manual_layout = qt.QVBoxLayout(manual_group)
        scope_form = qt.QFormLayout()
        self.deck = qt.QComboBox()
        self.deck.addItem("All decks", "")
        selected_deck = str(self.original.get("deck", ""))
        deck_names = available_deck_names(mw.col.decks)
        for deck_name in deck_names:
            self.deck.addItem(deck_name, deck_name)
        deck_index = self.deck.findData(selected_deck)
        if selected_deck and deck_index < 0:
            self.deck.addItem(f"Missing deck: {selected_deck}", selected_deck)
            deck_index = self.deck.count() - 1
        self.deck.setCurrentIndex(max(0, deck_index))
        self.deck.setToolTip(
            "A parent deck includes its subdecks. Choose All decks to use only "
            "the optional advanced search."
        )

        self.output = qt.QComboBox()
        self.output.addItem("MP3 — best Anki compatibility (recommended)", "mp3")
        self.output.addItem(
            "Keep source extension — adjusted audio is re-encoded",
            "source",
        )
        output_index = self.output.findData(
            str(self.original.get("output_format", "mp3"))
        )
        self.output.setCurrentIndex(max(0, output_index))
        self.output.setToolTip(
            "Keeping the extension is not necessarily lossless. Adjusted MP3, "
            "Opus, Ogg, AAC, M4A, and WebM audio is lossy re-encoded; Ogg output "
            "uses Opus, WAV output uses 16-bit PCM, and FLAC remains lossless. "
            "The retained original is never overwritten."
        )
        scope_form.addRow("Deck:", self.deck)
        scope_form.addRow("Output:", self.output)
        manual_layout.addLayout(scope_form)

        self.safety = qt.QLabel()
        self.safety.setWordWrap(True)
        manual_layout.addWidget(self.safety)

        self.last_run = qt.QLabel(_last_report_summary())
        self.last_run.setWordWrap(True)
        manual_layout.addWidget(self.last_run)

        layout.addWidget(manual_group)

        self.advanced_button = qt.QToolButton()
        self.advanced_button.setText("More settings")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setChecked(False)
        self.advanced_button.setToolButtonStyle(
            qt.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_button.setArrowType(qt.Qt.ArrowType.RightArrow)
        layout.addWidget(self.advanced_button)

        self.advanced_panel = qt.QWidget()
        advanced = qt.QFormLayout()
        self.query = qt.QLineEdit(str(self.original.get("query", "")))
        self.query.setPlaceholderText(
            'Leave blank for the whole deck; for example, tag:listening'
        )
        self.query.setToolTip(
            "Optional Anki Browser search terms used to narrow the selected "
            "deck. Examples: tag:listening, note:\"Sentence\", or -tag:music."
        )
        self.effective_query = qt.QLineEdit()
        self.effective_query.setReadOnly(True)
        self.effective_query.setToolTip(
            "The exact combined search Anki will use: selected deck plus the "
            "optional narrowing search above."
        )
        self.preset = qt.QComboBox()
        self.preset.addItem("Study speech — -18 LUFS (recommended)", -18.0)
        self.preset.addItem("Quieter — -20 LUFS", -20.0)
        self.preset.addItem("Podcast reference — -16 LUFS (Apple)", -16.0)
        self.preset.addItem("Broadcast reference — -23 LUFS (EBU R128)", -23.0)
        self.preset.setToolTip(
            "-18 LUFS is the add-on's study-speech default. -16 LUFS follows "
            "Apple's podcast guidance; -23 LUFS is the EBU R128 broadcast target."
        )
        preset_index = self.preset.findData(float(self.original["target_lufs"]))
        self.preset.setCurrentIndex(max(0, preset_index))
        self.boost = qt.QComboBox()
        self.boost.addItem("Downward only — never boost", 0.0)
        self.boost.addItem("Gentle — up to 6 dB", 6.0)
        self.boost.addItem(
            "Uncapped — reach target unless peak-limited (recommended)",
            "uncapped",
        )
        stored_boost = self.original.get("max_boost_db")
        selected_boost = (
            float(stored_boost)
            if stored_boost is not None and float(stored_boost) in (0.0, 6.0)
            else "uncapped"
        )
        boost_index = self.boost.findData(selected_boost)
        self.boost.setCurrentIndex(max(0, boost_index))
        self.boost.setToolTip(
            "Attenuation is always allowed as needed. This setting controls only "
            "increases; the true-peak ceiling still limits every boost."
        )
        self.automatic_display = qt.QComboBox()
        self.automatic_display.addItem(
            "Brief bottom notification (recommended)",
            "compact",
        )
        self.automatic_display.addItem(
            "Detailed progress window",
            "window",
        )
        display_index = self.automatic_display.findData(
            str(self.original.get("automatic_display", "compact"))
        )
        self.automatic_display.setCurrentIndex(max(0, display_index))
        self.automatic_display.setToolTip(
            "Manual runs always show detailed progress. Automatic runs can "
            "use a temporary Anki notification or open the full log window."
        )

        def update_safety_text(*_args):
            boost_data = self.boost.currentData()
            boost_text = (
                "never boosts quiet audio"
                if boost_data == 0.0
                else (
                    "boosts as needed, subject to the peak ceiling"
                    if boost_data == "uncapped"
                    else f"at most +{float(boost_data):.0f} dB boost"
                )
            )
            self.safety.setText(
                f"Settings: {float(self.preset.currentData()):.0f} "
                f"LUFS, -2 dBTP peak ceiling, and {boost_text}."
            )

        self.preset.currentIndexChanged.connect(update_safety_text)
        self.boost.currentIndexChanged.connect(update_safety_text)
        update_safety_text()
        self.on_startup = qt.QCheckBox("Anki starts")
        self.on_startup.setChecked(bool(self.original["startup"]))
        self.after_sync = qt.QCheckBox("A sync finishes")
        self.after_sync.setChecked(bool(self.original["after_sync"]))
        self.after_changes = qt.QCheckBox("Notes are added or edited")
        self.after_changes.setChecked(bool(self.original["after_changes"]))
        search_help = qt.QLabel(
            "Optionally narrow the chosen deck using the same search syntax as "
            "Anki's Browser. The final scope below shows exactly what will run."
        )
        search_help.setWordWrap(True)
        advanced.addRow(search_help)
        advanced.addRow("Narrow with Anki search:", self.query)
        advanced.addRow("Final scope:", self.effective_query)
        advanced.addRow("Target loudness:", self.preset)
        advanced.addRow("Maximum loudness increase:", self.boost)
        self.advanced_panel.setLayout(advanced)
        self.advanced_panel.setVisible(False)
        layout.addWidget(self.advanced_panel)

        self.automation_section = qt.QWidget()
        automation_layout = qt.QVBoxLayout(self.automation_section)
        automation_layout.setContentsMargins(0, 0, 0, 0)
        automation_header = qt.QWidget()
        automation_header_layout = qt.QHBoxLayout(automation_header)
        automation_header_layout.setContentsMargins(0, 0, 0, 0)
        self.automation_button = qt.QToolButton()
        self.automation_button.setText("Run automatically")
        self.automation_button.setCheckable(True)
        self.automation_button.setChecked(False)
        self.automation_button.setToolButtonStyle(
            qt.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.automation_button.setArrowType(qt.Qt.ArrowType.RightArrow)
        self.enabled = qt.QCheckBox("Enabled")
        self.enabled.setChecked(bool(self.original["enabled"]))
        automation_header_layout.addWidget(self.automation_button)
        automation_header_layout.addStretch()
        automation_header_layout.addWidget(self.enabled)
        automation_layout.addWidget(automation_header)

        self.automatic_options = qt.QWidget()
        automatic_options_layout = qt.QFormLayout(self.automatic_options)
        automatic_options_layout.setContentsMargins(20, 0, 0, 0)
        automatic_description = qt.QLabel(
            "Uses the deck, filters, output, and loudness settings above."
        )
        automatic_description.setWordWrap(True)
        automatic_options_layout.addRow(automatic_description)
        run_when = qt.QWidget()
        run_when_layout = qt.QVBoxLayout(run_when)
        run_when_layout.setContentsMargins(0, 0, 0, 0)
        run_when_layout.addWidget(self.on_startup)
        run_when_layout.addWidget(self.after_sync)
        run_when_layout.addWidget(self.after_changes)
        automatic_options_layout.addRow("Run when:", run_when)
        automatic_options_layout.addRow("Show:", self.automatic_display)
        self.automatic_options.setEnabled(self.enabled.isChecked())
        self.automatic_options.setVisible(False)
        self.enabled.toggled.connect(self.automatic_options.setEnabled)
        automation_layout.addWidget(self.automatic_options)
        layout.addWidget(self.automation_section)

        def update_effective_query(*_args):
            combined = effective_search(
                str(self.deck.currentData() or ""),
                self.query.text(),
            )
            self.effective_query.setText(combined or "Entire collection")

        self.deck.currentIndexChanged.connect(update_effective_query)
        self.query.textChanged.connect(update_effective_query)
        update_effective_query()

        def toggle_advanced(visible):
            self.advanced_panel.setVisible(visible)
            self.advanced_button.setArrowType(
                qt.Qt.ArrowType.DownArrow if visible else qt.Qt.ArrowType.RightArrow
            )
            self.adjustSize()

        self.advanced_button.toggled.connect(toggle_advanced)

        def toggle_automation(visible):
            self.automatic_options.setVisible(visible)
            self.automation_button.setArrowType(
                qt.Qt.ArrowType.DownArrow if visible else qt.Qt.ArrowType.RightArrow
            )
            self.adjustSize()

        self.automation_button.toggled.connect(toggle_automation)

        self.preview_button = qt.QPushButton("Preview")
        self.revert_button = qt.QPushButton("Restore")
        self.report_button = qt.QPushButton("Progress / last run")
        self.run_button = qt.QPushButton("Save and normalize")
        self.run_button.setDefault(True)
        self.preview_button.clicked.connect(self.preview)
        self.run_button.clicked.connect(self.run_now)
        self.revert_button.clicked.connect(self.revert_to_originals)
        self.report_button.clicked.connect(self.show_last_report)
        self.preview_button.setToolTip(
            f"Inspect up to {PREVIEW_SAMPLE_LIMIT} files spread across the scope "
            "without changing anything."
        )
        self.revert_button.setToolTip(
            "Restore matching references to retained originals. Generated copies "
            "are kept, automatic runs are disabled, and no exclusion tag is added."
        )
        self.run_button.setToolTip(
            "Save these settings and normalize every matching audio file."
        )
        action_layout = qt.QHBoxLayout()
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.revert_button)
        action_layout.addWidget(self.report_button)
        action_layout.addStretch()
        action_layout.addWidget(self.run_button)
        layout.addStretch()
        layout.addLayout(action_layout)
        self.setLayout(layout)
        self.set_external_scope(external_scope)

    def set_external_scope(self, external_scope):
        """Apply or clear a one-off Browser scope without saving it as automation."""

        self.external_scope = external_scope
        if external_scope is None:
            self.setWindowTitle("Normalize Card Audio")
            self.scope_notice.setVisible(False)
            self.deck.setEnabled(True)
            deck_index = self.deck.findData(str(self.original.get("deck", "")))
            self.deck.setCurrentIndex(max(0, deck_index))
            self.query.setReadOnly(False)
            self.query.setText(str(self.original.get("query", "")))
            self.automation_section.setVisible(True)
            self.revert_button.setText("Restore")
            self.run_button.setText("Save and normalize")
            return

        query, label = external_scope
        self.setWindowTitle("Normalize Browser Audio")
        self.scope_notice.setText(
            f"<b>Browser scope:</b> {html.escape(str(label))}<br>"
            "The actions in this window affect only this fixed Browser scope. "
            "Your saved automatic-run scope will not be replaced."
        )
        self.scope_notice.setVisible(True)
        self.deck.setCurrentIndex(0)
        self.deck.setEnabled(False)
        self.query.setText(str(query))
        self.query.setReadOnly(True)
        self.automation_section.setVisible(False)
        self.revert_button.setText("Restore scoped notes")
        self.run_button.setText("Normalize scoped notes")

    def settings_to_persist(self, settings):
        """Keep a temporary Browser scope out of the saved automatic settings."""

        if self.external_scope is None:
            return dict(settings)
        persistent = dict(settings)
        for key in (
            "deck",
            "query",
            "enabled",
            "startup",
            "after_sync",
            "after_changes",
            "automatic_display",
        ):
            persistent[key] = self.original[key]
        return persistent

    def values(self):
        boost_data = self.boost.currentData()
        return {
            "enabled": self.enabled.isChecked(),
            "deck": str(self.deck.currentData() or ""),
            "query": self.query.text().strip(),
            "output_format": str(self.output.currentData()),
            "target_lufs": float(self.preset.currentData()),
            "max_boost_db": (
                None if boost_data == "uncapped" else float(boost_data)
            ),
            "startup": self.on_startup.isChecked(),
            "after_sync": self.after_sync.isChecked(),
            "after_changes": self.after_changes.isChecked(),
            "automatic_display": str(self.automatic_display.currentData()),
        }

    def _validate(self):
        settings = self.values()
        try:
            recipe_from_settings(settings).validate()
            mw.col.find_notes(search_from_settings(settings))
        except Exception as error:
            qt.QMessageBox.critical(self, "Invalid settings", str(error))
            return None
        return settings

    def preview(self):
        settings = self._validate()
        if settings is None:
            return
        self.preview_button.setEnabled(False)
        self.preview_button.setText("Scanning notes…")
        dialog_id = id(self)
        profile_name = _profile_name()

        def current_dialog():
            dialog = getattr(mw, "_audio_normalizer_settings_dialog", None)
            return dialog if dialog is not None and id(dialog) == dialog_id else None

        def restore_button():
            dialog = current_dialog()
            if dialog is not None:
                dialog.preview_button.setEnabled(True)
                dialog.preview_button.setText("Preview")
            return dialog

        def success(text):
            dialog = restore_button()
            if dialog is None or _profile_name() != profile_name:
                return
            qt.QMessageBox.information(dialog, "Preview", text)

        def failure(error):
            dialog = restore_button()
            if dialog is None or _profile_name() != profile_name:
                return
            qt.QMessageBox.critical(dialog, "Check failed", str(error))

        def set_button_text(text):
            def update_button():
                dialog = current_dialog()
                if dialog is not None and _profile_name() == profile_name:
                    dialog.preview_button.setText(text)

            mw.taskman.run_on_main(update_button)

        scanned_notes = 0

        def scanning_progress():
            nonlocal scanned_notes
            scanned_notes += 250
            set_button_text(f"Scanning notes ({scanned_notes:,}+)…")

        def measure(preview_scope):
            if current_dialog() is None or _profile_name() != profile_name:
                return

            def update_progress(completed, total):
                if completed == 0:
                    set_button_text("Preparing FFmpeg…")
                else:
                    set_button_text(f"Checking {completed}/{total}…")

            try:
                QueryOp(
                    parent=mw,
                    op=lambda _collection: _preview_text(
                        settings,
                        preview_scope,
                        progress_callback=update_progress,
                    ),
                    success=success,
                ).without_collection().failure(failure).run_in_background()
            except Exception as error:
                failure(error)

        try:
            QueryOp(
                parent=mw,
                op=lambda collection: _preview_scope(
                    settings,
                    collection,
                    progress_callback=scanning_progress,
                ),
                success=measure,
            ).failure(failure).run_in_background()
        except Exception as error:
            failure(error)

    def run_now(self):
        settings = self._validate()
        if settings is None:
            return
        save_settings(self.settings_to_persist(settings))
        self.accept()
        schedule_run(
            "manual",
            0,
            allow_disabled=True,
            settings_override=(settings if self.external_scope is not None else None),
        )

    def show_last_report(self):
        if is_running():
            _status_dialog()._show(activate=True)
            return
        try:
            with open(_report_path(), "r", encoding="utf-8") as report:
                text = report.read()
        except OSError:
            text = "No processing run has completed yet."
        _status_dialog().show_report(text)

    def revert_to_originals(self):
        settings = self._validate()
        if settings is None:
            return
        try:
            _note_ids, discovery = _discover(
                settings,
                progress_callback=mw.app.processEvents,
            )
            plan = plan_revert(
                mw.col.media.dir(),
                discovery.filenames,
                ProcessingState(_state_path()),
            )
        except Exception as error:
            qt.QMessageBox.critical(
                self,
                "Could not plan audio restoration",
                str(error),
            )
            return

        notes_to_change = {
            reference.note_id
            for reference in discovery.references
            if reference.filename in plan.replacements
        }
        scope = search_from_settings(settings) or "entire collection"
        summary = (
            f"Scope: {scope}\n\n"
            f"Original audio files that can be restored: "
            f"{len(plan.replacements)}\n"
            f"Notes that will change: {len(notes_to_change)}\n"
            f"References already using originals: {plan.already_original}\n"
            f"Generated files with no unambiguous original: "
            f"{len(plan.unresolved)}"
        )
        if not plan.replacements:
            qt.QMessageBox.information(
                self,
                "Restore matching audio",
                summary
                + "\n\nNo note references can be restored automatically.",
            )
            return

        answer = qt.QMessageBox.question(
            self,
            "Restore matching audio from originals?",
            summary
            + "\n\n"
            "This changes note references only. Generated audio files are "
            "kept for recovery and can be cleaned up later after verification.\n\n"
            "Automatic normalization for this profile will be disabled. No tag "
            "or permanent exclusion is added; if you later enable automatic "
            "runs with a scope containing these notes, they can be normalized "
            "again.",
            qt.QMessageBox.StandardButton.Yes
            | qt.QMessageBox.StandardButton.Cancel,
            qt.QMessageBox.StandardButton.Cancel,
        )
        if answer != qt.QMessageBox.StandardButton.Yes:
            return

        settings["enabled"] = False
        persistent = self.settings_to_persist(settings)
        persistent["enabled"] = False
        save_settings(persistent)
        self.original = dict(persistent)
        self.enabled.setChecked(False)
        collection = mw.col
        operation_result = {"updated_notes": 0}
        dialog_id = id(self)
        restore_button_text = (
            "Restore scoped notes" if self.external_scope else "Restore"
        )
        self.revert_button.setEnabled(False)
        self.revert_button.setText("Restoring…")

        def current_dialog():
            dialog = getattr(mw, "_audio_normalizer_settings_dialog", None)
            return dialog if dialog is not None and id(dialog) == dialog_id else None

        def restore_references(active_collection):
            if active_collection is not collection:
                raise RuntimeError(
                    "The active Anki profile changed before notes were restored."
                )
            operation_result["updated_notes"] = replace_media_references(
                active_collection,
                discovery.references,
                plan.replacements,
                undo_label="Restore original card audio",
            )
            changes = OpChanges()
            changes.note = bool(operation_result["updated_notes"])
            return changes

        def restore_finished(_changes):
            updated_notes = operation_result["updated_notes"]
            _write_revert_report(plan, updated_notes, scope)
            dialog = current_dialog()
            if dialog is not None:
                dialog.revert_button.setEnabled(True)
                dialog.revert_button.setText(restore_button_text)
                dialog.last_run.setText(_last_report_summary())
            if updated_notes:
                mw.reset()
            tooltip(
                "Card Audio Normalizer — restored "
                f"{len(plan.replacements)} unique audio files across "
                f"{updated_notes} notes; automatic normalization disabled.",
                period=7000,
            )

        def restore_failed(error):
            dialog = current_dialog()
            if dialog is not None:
                dialog.revert_button.setEnabled(True)
                dialog.revert_button.setText(restore_button_text)
            qt.QMessageBox.critical(dialog or mw, "Restore failed", str(error))

        CollectionOp(parent=mw, op=restore_references).success(
            restore_finished
        ).failure(restore_failed).run_in_background(
            initiator=_OPERATION_INITIATOR
        )

def show_dialog(external_scope=None):
    existing = getattr(mw, "_audio_normalizer_settings_dialog", None)
    if existing is not None:
        existing.set_external_scope(external_scope)
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return

    dialog = CardAudioNormalizerDialog(mw, external_scope=external_scope)
    mw._audio_normalizer_settings_dialog = dialog

    def cleanup(*_args):
        if getattr(mw, "_audio_normalizer_settings_dialog", None) is dialog:
            mw._audio_normalizer_settings_dialog = None
        dialog.deleteLater()

    dialog.finished.connect(cleanup)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _browser_search(browser):
    """Return the executed Browser search, falling back to the visible text."""

    search = getattr(browser, "_lastSearchTxt", None)
    if not isinstance(search, str):
        search = browser.current_search()
    return str(search or "").replace("\n", " ").strip()


def show_browser_scope(browser, selected_only):
    try:
        selected = list(browser.selected_notes()) if selected_only else []
        if selected_only and not selected:
            qt.QMessageBox.information(
                browser,
                "No notes selected",
                "Select one or more Browser rows first.",
            )
            return
        current_search = _browser_search(browser)
        query = browser_scope_search(current_search, selected)
    except Exception as error:
        qt.QMessageBox.critical(
            browser,
            "Could not use Browser scope",
            str(error),
        )
        return

    if selected:
        count = len(set(int(note_id) for note_id in selected))
        label = f"{count} selected note{'s' if count != 1 else ''} from the current results"
    else:
        label = f'Current Browser search: {current_search or "entire collection"}'
    show_dialog((query, label))


def setup_browser_menu(browser):
    if getattr(browser, "_audio_normalizer_actions", None) is not None:
        return
    selected_action = qt.QAction(
        "Audio Normalizer — Selected Notes…",
        browser,
    )
    selected_action.triggered.connect(
        lambda _checked=False: show_browser_scope(browser, selected_only=True)
    )
    search_action = qt.QAction(
        "Audio Normalizer — Current Search…",
        browser,
    )
    search_action.triggered.connect(
        lambda _checked=False: show_browser_scope(browser, selected_only=False)
    )
    browser.form.menu_Notes.addSeparator()
    browser.form.menu_Notes.addAction(selected_action)
    browser.form.menu_Notes.addAction(search_action)
    browser._audio_normalizer_actions = (selected_action, search_action)


def add_browser_context_menu(browser, menu):
    try:
        count = len(set(int(note_id) for note_id in browser.selected_notes()))
    except Exception:
        return
    if not count:
        return
    menu.addSeparator()
    label = (
        "Open note in Audio Normalizer…"
        if count == 1
        else f"Open {count} notes in Audio Normalizer…"
    )
    action = menu.addAction(label)
    action.triggered.connect(
        lambda _checked=False: show_browser_scope(browser, selected_only=True)
    )


def _finish_run(
    outcome,
    reason,
    collection,
    profile_name,
    automatic_display,
    error=None,
):
    if _media_sync_running:
        qt.QTimer.singleShot(
            500,
            lambda: _finish_run(
                outcome,
                reason,
                collection,
                profile_name,
                automatic_display,
                error,
            ),
        )
        return
    try:
        if error is not None:
            raise error
        discovery, batch = outcome
        if mw.col is not collection or _profile_name() != profile_name:
            raise RuntimeError("The active Anki profile changed during the run.")
        total_files = len(discovery.filenames)
        if not total_files:
            _complete_run(
                discovery,
                batch,
                0,
                reason,
                collection,
                profile_name,
                automatic_display,
            )
            return

        if not batch.replacements:
            _complete_run(
                discovery,
                batch,
                0,
                reason,
                collection,
                profile_name,
                automatic_display,
            )
            return

        operation_result = {"updated_notes": 0}

        def update_references(active_collection):
            if active_collection is not collection:
                raise RuntimeError(
                    "The active Anki profile changed before notes were updated."
                )
            operation_result["updated_notes"] = replace_media_references(
                active_collection,
                discovery.references,
                batch.replacements,
            )
            changes = OpChanges()
            changes.note = bool(operation_result["updated_notes"])
            return changes

        def updated(_changes):
            _complete_run(
                discovery,
                batch,
                operation_result["updated_notes"],
                reason,
                collection,
                profile_name,
                automatic_display,
            )

        def update_failed(update_error):
            _complete_run(
                discovery,
                batch,
                0,
                reason,
                collection,
                profile_name,
                automatic_display,
                error=update_error,
            )

        CollectionOp(parent=mw, op=update_references).success(updated).failure(
            update_failed
        ).run_in_background(initiator=_OPERATION_INITIATOR)
    except Exception as error:
        _complete_run(
            None,
            None,
            0,
            reason,
            collection,
            profile_name,
            automatic_display,
            error=error,
        )


def _complete_run(
    discovery,
    batch,
    updated_notes,
    reason,
    collection,
    profile_name,
    automatic_display,
    error=None,
):
    global _running, _pending, _active_cancel_event, _pending_settings_override
    try:
        if error is not None:
            raise error
        if mw.col is not collection or _profile_name() != profile_name:
            raise RuntimeError("The active Anki profile changed during the run.")
        total_files = len(discovery.filenames)
        if not total_files:
            _status_dialog().finish_run(batch, 0, 0)
            if reason == "manual":
                tooltip(
                    "Card Audio Normalizer — no matching audio found.",
                    period=4000,
                )
            elif automatic_display == "compact":
                tooltip(
                    "Card Audio Normalizer — 0 matching audio files.",
                    period=5000,
                )
            return
        if updated_notes:
            mw.reset()
        _write_report(batch, updated_notes, reason)
        _status_dialog().finish_run(batch, updated_notes, total_files)
        if reason == "manual" or automatic_display == "compact":
            run_state = "stopped; " if batch.cancelled else ""
            unchanged = batch.count(
                "already_processed",
                "already_ready",
                "already_ready_cached",
            )
            tooltip(
                "Card Audio Normalizer — "
                + run_state
                + f"{len(batch.files)} checked: "
                f"{batch.changed} new files, {unchanged} unchanged, "
                f"{batch.reused} reused; {updated_notes} notes updated, "
                f"{batch.warnings} warnings, {batch.errors} errors.",
                period=7000,
            )
    except Exception as error:
        show_window = reason == "manual" or automatic_display == "window"
        _status_dialog().show_error(
            str(error),
            activate=reason == "manual",
            show=show_window,
        )
        if not show_window:
            tooltip(
                "Card Audio Normalizer failed — " + str(error)[:300],
                period=8000,
            )
    finally:
        _running = False
        _active_cancel_event = None
        if _pending:
            settings_override = _pending_settings_override
            _pending = False
            _pending_settings_override = None
            schedule_run(
                "manual" if settings_override is not None else "pending",
                500,
                allow_disabled=settings_override is not None,
                settings_override=settings_override,
            )


def _start_run(reason, allow_disabled=False, settings_override=None):
    global _running, _pending, _active_cancel_event, _pending_settings_override
    settings = dict(settings_override) if settings_override is not None else load_settings()
    automatic_display = str(
        settings.get("automatic_display", "compact")
    )
    if not settings["enabled"] and not allow_disabled:
        return
    if mw.col is None:
        return
    if _media_sync_running:
        schedule_run(
            reason,
            500,
            allow_disabled=allow_disabled,
            settings_override=settings_override,
        )
        return
    if _running:
        _pending = True
        if settings_override is not None:
            _pending_settings_override = dict(settings_override)
        return
    _running = True
    if reason == "manual":
        tooltip("Card Audio Normalizer — scanning matching audio…", period=3000)
    recipe = recipe_from_settings(settings)
    collection = mw.col
    profile_name = _profile_name()
    media_directory = collection.media.dir()
    state_path = _state_path()
    status_dialog = _status_dialog()
    cancel_event = threading.Event()
    _active_cancel_event = cancel_event
    progress_relay = ProgressRelay(status_dialog)
    status_dialog.begin_run(
        reason,
        cancel_event.set,
        automatic_display=automatic_display,
    )

    def discover(active_collection):
        if active_collection is not collection:
            raise RuntimeError(
                "The active Anki profile changed before the run started."
            )
        _note_ids, discovery = _discover(
            settings,
            collection=active_collection,
        )
        return discovery

    def process(discovery):
        backend = _ensure_backend()
        state = ProcessingState(state_path)
        service = AudioNormalizationService(
            backend, media_directory, recipe, state=state
        )
        batch = service.process(
            discovery.filenames,
            progress_callback=progress_relay.push,
            should_cancel=cancel_event.is_set,
        )
        return batch

    def done(outcome):
        progress_relay.finish()
        _finish_run(
            outcome,
            reason,
            collection,
            profile_name,
            automatic_display,
        )

    def failed(error):
        progress_relay.finish()
        _finish_run(
            None,
            reason,
            collection,
            profile_name,
            automatic_display,
            error=error,
        )

    def start_processing(discovery):
        total_files = len(discovery.filenames)
        status_dialog.set_total(total_files, reason)
        if not total_files:
            done((discovery, BatchResult()))
            return
        QueryOp(
            parent=mw,
            op=lambda _collection: process(discovery),
            success=lambda batch: done((discovery, batch)),
        ).without_collection().failure(failed).run_in_background()

    try:
        QueryOp(parent=mw, op=discover, success=start_processing).failure(
            failed
        ).run_in_background()
    except Exception:
        _running = False
        raise


def schedule_run(reason, delay_ms=0, allow_disabled=False, settings_override=None):
    global _scheduled, _scheduled_allow_disabled, _pending, _schedule_generation
    global _pending_settings_override
    settings = dict(settings_override) if settings_override is not None else load_settings()
    if not settings["enabled"] and not allow_disabled:
        return
    if _running:
        _pending = True
        if settings_override is not None:
            _pending_settings_override = dict(settings_override)
        return
    _scheduled_allow_disabled = _scheduled_allow_disabled or allow_disabled
    if _scheduled:
        if reason == "manual":
            # Do not let a delayed startup/change check swallow an explicit click.
            _schedule_generation += 1
            _scheduled = False
            schedule_run(
                reason,
                0,
                allow_disabled=True,
                settings_override=settings_override,
            )
        return
    _scheduled = True
    _schedule_generation += 1
    generation = _schedule_generation

    def dispatch():
        global _scheduled, _scheduled_allow_disabled
        if generation != _schedule_generation:
            return
        permitted = _scheduled_allow_disabled
        _scheduled = False
        _scheduled_allow_disabled = False
        _start_run(reason, permitted, settings_override=settings_override)

    qt.QTimer.singleShot(delay_ms, dispatch)


def is_running():
    return _running


def setup_menu():
    if getattr(mw, "_audio_normalizer_action", None) is not None:
        return
    action = qt.QAction("Normalize Card Audio…", mw)
    action.triggered.connect(lambda _checked=False: show_dialog())
    mw.form.menuTools.addAction(action)
    mw._audio_normalizer_action = action
    mw.addonManager.setConfigAction(_addon_module(), show_dialog)


def on_profile_did_open():
    settings = load_settings()
    if settings["enabled"] and settings["startup"]:
        schedule_run("startup", 2500)


def on_profile_will_close():
    global _scheduled, _scheduled_allow_disabled, _pending
    global _post_sync_waiting, _schedule_generation, _pending_settings_override
    if _active_cancel_event is not None:
        _active_cancel_event.set()
    _schedule_generation += 1
    _scheduled = False
    _scheduled_allow_disabled = False
    _pending = False
    _pending_settings_override = None
    _post_sync_waiting = False


def on_sync_did_finish():
    global _post_sync_waiting
    settings = load_settings()
    if settings["enabled"] and settings["after_sync"]:
        _post_sync_waiting = True
        qt.QTimer.singleShot(1000, _dispatch_post_sync_if_ready)


def _dispatch_post_sync_if_ready():
    global _post_sync_waiting
    if _post_sync_waiting and not _media_sync_running:
        _post_sync_waiting = False
        schedule_run("after_sync", 500)


def on_media_sync_did_start_or_stop(running):
    global _media_sync_running
    _media_sync_running = running
    if not running:
        _dispatch_post_sync_if_ready()


def on_operation_did_execute(changes, handler):
    if _updating_note_references or handler is _OPERATION_INITIATOR:
        return
    settings = load_settings()
    if (
        settings["enabled"]
        and settings["after_changes"]
        and (getattr(changes, "note", False) or getattr(changes, "note_text", False))
    ):
        schedule_run("note_change", 2500)


if hasattr(gui_hooks, "main_window_did_init"):
    gui_hooks.main_window_did_init.append(setup_menu)
else:
    setup_menu()
if hasattr(gui_hooks, "profile_did_open"):
    gui_hooks.profile_did_open.append(on_profile_did_open)
if hasattr(gui_hooks, "profile_will_close"):
    gui_hooks.profile_will_close.append(on_profile_will_close)
if hasattr(gui_hooks, "sync_did_finish"):
    gui_hooks.sync_did_finish.append(on_sync_did_finish)
if hasattr(gui_hooks, "media_sync_did_start_or_stop"):
    gui_hooks.media_sync_did_start_or_stop.append(on_media_sync_did_start_or_stop)
if hasattr(gui_hooks, "operation_did_execute"):
    gui_hooks.operation_did_execute.append(on_operation_did_execute)
if hasattr(gui_hooks, "browser_menus_did_init"):
    gui_hooks.browser_menus_did_init.append(setup_browser_menu)
if hasattr(gui_hooks, "browser_will_show_context_menu"):
    gui_hooks.browser_will_show_context_menu.append(add_browser_context_menu)
