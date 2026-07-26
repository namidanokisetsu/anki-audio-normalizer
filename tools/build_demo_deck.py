#!/usr/bin/env python3
"""Build a three-card .apkg for demonstrating Audio Normalizer."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "audio-normalizer-demo.apkg"
SOURCE_URL = (
    "https://commons.wikimedia.org/wiki/"
    "Special:Redirect/file/Jimmy_Wales_voice.ogg"
)
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Jimmy_Wales_voice.ogg"
SOURCE_SHA256 = "a5a1f939cf0514bec3709db1e4b6cb36c60a17ae359394a81d92ee5f3ca65087"
MAX_SOURCE_BYTES = 5 * 1024 * 1024
EXPECTED_MEDIA = {
    "demo_speech_too_quiet.opus",
    "demo_speech_normal.mp3",
    "demo_speech_too_loud.mp3",
}


def require_dependencies():
    try:
        import genanki
    except ImportError as error:
        raise SystemExit(
            "This builder needs genanki. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from error

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("This builder needs FFmpeg on PATH.")
    return genanki, ffmpeg


def download_source(destination: Path) -> None:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Audio-Normalizer-demo-deck/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise RuntimeError("The speech source exceeded the expected size limit.")
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            "The speech source changed unexpectedly; refusing to build an "
            f"unverified deck (received SHA-256 {digest})."
        )
    destination.write_bytes(data)


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(stderr.strip().splitlines()[-1])


def measurement_text(measurement) -> str:
    return (
        f"{measurement.integrated_lufs:.2f} LUFS; "
        f"true peak {measurement.true_peak_dbtp:.2f} dBTP"
    )


def expected_text(condition: str, decision) -> str:
    if condition == "Too quiet":
        return (
            f"The default Uncapped setting raises it by {decision.gain_db:.2f} "
            "dB to the target and converts it to MP3."
        )
    if condition == "Already on target":
        return (
            "It is already within the default ±0.5 LU tolerance, so the file "
            "and note should stay unchanged."
        )
    return (
        f"With the default settings, lower it by {abs(decision.gain_db):.2f} dB "
        "and replace the reference with a normalized MP3."
    )


def build(output: Path) -> list[tuple[str, str, str]]:
    genanki, ffmpeg = require_dependencies()
    sys.path.insert(0, str(ROOT))
    from normalization.core import ProcessingRecipe, decide_gain
    from normalization.ffmpeg_backend import FFmpegBackend

    recipe = ProcessingRecipe()
    backend = FFmpegBackend(ffmpeg)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="audio_normalizer_demo_") as temporary:
        working = Path(temporary)
        source = working / "source.ogg"
        conditioned = working / "conditioned.wav"
        reference = working / "reference.wav"
        download_source(source)

        # Compress the source once to leave enough peak headroom for a genuinely
        # loud +6 dB variant. Every card still contains the same complete speech.
        run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-nostats",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                "acompressor=threshold=0.0316:ratio=8:attack=1:release=150:makeup=1",
                "-codec:a",
                "pcm_s16le",
                str(conditioned),
            ]
        )
        conditioned_measurement = backend.analyze(str(conditioned), recipe)
        if not conditioned_measurement.is_usable:
            raise RuntimeError("The conditioned source could not be measured.")
        # The matched reference needs enough headroom for an obviously loud
        # variant. Find a gain that reaches the target while a limiter used only
        # during fixture construction holds the reference near -11 dBTP.
        lower_gain, upper_gain = 0.0, 60.0
        for _attempt in range(12):
            candidate_gain = (lower_gain + upper_gain) / 2
            run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-nostats",
                    "-i",
                    str(conditioned),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    (
                        f"volume={candidate_gain:.3f}dB,"
                        "alimiter=limit=0.251189:level=false:attack=1:release=50"
                    ),
                    "-codec:a",
                    "pcm_s16le",
                    str(reference),
                ]
            )
            reference_measurement = backend.analyze(str(reference), recipe)
            if not reference_measurement.is_usable:
                raise RuntimeError("The reference fixture could not be measured.")
            if reference_measurement.integrated_lufs < recipe.target_lufs:
                lower_gain = candidate_gain
            else:
                upper_gain = candidate_gain
        if abs(reference_measurement.integrated_lufs - recipe.target_lufs) > 0.1:
            raise RuntimeError("The reference fixture did not reach its target.")

        specifications = (
            (
                "Too quiet",
                "demo_speech_too_quiet.opus",
                "Opus",
                -9.0,
                "About 10 dB quieter than the add-on's default target.",
            ),
            (
                "Already on target",
                "demo_speech_normal.mp3",
                "MP3",
                0.0,
                "Already at the add-on's default target.",
            ),
            (
                "Too loud",
                "demo_speech_too_loud.mp3",
                "MP3",
                9.0,
                "About 9 dB louder than the add-on's default target.",
            ),
        )

        built = []
        for condition, filename, audio_format, gain_db, explanation in specifications:
            path = working / filename
            backend.transform(
                str(reference),
                str(path),
                gain_db,
                recipe,
                output_format=path.suffix,
            )
            measurement = backend.analyze(str(path), recipe)
            if not measurement.is_usable:
                raise RuntimeError(f"Could not measure generated file: {filename}")
            decision = decide_gain(
                measurement,
                recipe,
                needs_conversion=path.suffix.lower() != ".mp3",
            )
            built.append(
                {
                    "condition": condition,
                    "filename": filename,
                    "format": audio_format,
                    "explanation": explanation,
                    "measurement": measurement_text(measurement),
                    "expected": expected_text(condition, decision),
                    "audio": f"[sound:{filename}]",
                    "path": path,
                }
            )

        model = genanki.Model(
            1607392319,
            "Audio Normalizer Demo",
            fields=[
                {"name": "Condition"},
                {"name": "Original filename"},
                {"name": "Current filename"},
                {"name": "Format"},
                {"name": "Explanation"},
                {"name": "Original measurement"},
                {"name": "Expected result"},
                {"name": "Audio"},
                {"name": "Source"},
            ],
            templates=[
                {
                    "name": "Listen and inspect",
                    "qfmt": """
<div class="condition">{{Condition}}</div>
<div class="filename-label">Original audio file</div>
<div class="filename">{{Original filename}}</div>
<div class="filename-label">Current audio file</div>
<div class="filename filename-current">{{Current filename}}</div>
<div class="detail"><b>Original format:</b> {{Format}}</div>
<div class="detail"><b>Original level:</b> {{Original measurement}}</div>
<div class="explanation">{{Explanation}}</div>
<div class="audio">{{Audio}}</div>
""",
                    "afmt": """
{{FrontSide}}
<hr id="answer">
<div class="expected"><b>Expected Audio Normalizer result</b><br>{{Expected result}}</div>
<div class="source">{{Source}}</div>
""",
                }
            ],
            css="""
.card { font-family: Arial, sans-serif; font-size: 18px; line-height: 1.45;
  text-align: left; color: #202124; background: #fff; max-width: 680px;
  margin: 30px auto; }
.condition { font-size: 30px; font-weight: 700; margin-bottom: 8px; }
.filename-label { margin-top: 12px; font-size: 13px; color: #666; }
.filename { font-family: monospace; overflow-wrap: anywhere; color: #455a64;
  margin-bottom: 8px; }
.filename-current { margin-bottom: 20px; }
.detail { margin: 6px 0; }
.explanation, .expected { margin-top: 18px; padding: 14px; background: #eef5ff;
  border-radius: 8px; }
.audio { margin-top: 22px; }
.source { margin-top: 22px; font-size: 13px; color: #666; }
.nightMode .card { color: #eee; background: #202124; }
.nightMode .explanation, .nightMode .expected { background: #26364a; }
""",
        )
        deck = genanki.Deck(2059400110, "Audio Normalizer Demo")
        deck.description = (
            "Three matched speech files for testing Audio Normalizer: too quiet, "
            "on target, and too loud."
        )
        source_credit = (
            'Speech: <a href="'
            + SOURCE_PAGE
            + '">“Jimmy Wales voice”</a>, recorded by Vera de Kok, CC0 1.0. '
            "All three files are level and format variants of the same recording."
        )
        for item in built:
            note = genanki.Note(
                model=model,
                guid=genanki.guid_for("audio-normalizer-demo", item["filename"]),
                fields=[
                    item["condition"],
                    html.escape(item["filename"]),
                    (
                        '<span data-audio-normalizer-filename>'
                        + html.escape(item["filename"])
                        + "</span>"
                    ),
                    item["format"],
                    item["explanation"],
                    item["measurement"],
                    item["expected"],
                    item["audio"],
                    source_credit,
                ],
                tags=["audio-normalizer-demo"],
            )
            deck.add_note(note)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".audio_normalizer_demo_", suffix=".apkg", dir=output.parent
        )
        os.close(descriptor)
        temporary_output = Path(temporary_name)
        try:
            package = genanki.Package(deck)
            package.media_files = [str(item["path"]) for item in built]
            package.write_to_file(str(temporary_output))
            validate_package(temporary_output)
            os.replace(temporary_output, output)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()

        return [
            (item["condition"], item["filename"], item["measurement"])
            for item in built
        ]


def validate_package(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise RuntimeError(f"The generated deck has a corrupt entry: {bad_file}")
        names = set(archive.namelist())
        if "media" not in names:
            raise RuntimeError("The generated deck has no media map.")
        media = json.loads(archive.read("media"))
        if set(media.values()) != EXPECTED_MEDIA:
            raise RuntimeError("The generated deck does not contain the expected media.")
        databases = {"collection.anki2", "collection.anki21"} & names
        if not databases:
            raise RuntimeError("The generated deck has no Anki collection database.")
        database_name = sorted(databases)[-1]
        with tempfile.TemporaryDirectory(prefix="audio_normalizer_validate_") as temp:
            archive.extract(database_name, temp)
            with sqlite3.connect(Path(temp) / database_name) as connection:
                note_count = connection.execute("SELECT count(*) FROM notes").fetchone()[0]
                card_count = connection.execute("SELECT count(*) FROM cards").fetchone()[0]
                note_fields = [
                    fields.split("\x1f")
                    for (fields,) in connection.execute("SELECT flds FROM notes")
                ]
                decks_json, configs_json, models_json = connection.execute(
                    "SELECT decks, dconf, models FROM col"
                ).fetchone()
            if (note_count, card_count) != (3, 3):
                raise RuntimeError(
                    "The generated deck must contain exactly three notes and cards."
                )
            original_filenames = {fields[1] for fields in note_fields}
            current_filenames = {
                fields[2].removeprefix(
                    "<span data-audio-normalizer-filename>"
                ).removesuffix("</span>")
                for fields in note_fields
            }
            if original_filenames != EXPECTED_MEDIA:
                raise RuntimeError(
                    "The original filename labels do not match the deck media."
                )
            if current_filenames != EXPECTED_MEDIA:
                raise RuntimeError(
                    "The current filename labels do not match the deck media."
                )
            decks = json.loads(decks_json)
            configs = json.loads(configs_json)
            demo_deck = decks[str(2059400110)]
            if not configs[str(demo_deck["conf"])].get("autoplay"):
                raise RuntimeError("The demo deck does not enable audio autoplay.")
            models = json.loads(models_json)
            demo_model = models[str(1607392319)]
            question_template = demo_model["tmpls"][0]["qfmt"]
            if "{{Audio}}" not in question_template:
                raise RuntimeError("The demo card front does not contain its audio.")
            if not all(
                field in question_template
                for field in ("{{Original filename}}", "{{Current filename}}")
            ):
                raise RuntimeError(
                    "The demo card front does not show both filename fields."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"deck path (default: {DEFAULT_OUTPUT.name})",
    )
    arguments = parser.parse_args()
    results = build(arguments.output)
    print(arguments.output.resolve())
    for condition, filename, measurement in results:
        print(f"{condition}: {filename} — {measurement}")


if __name__ == "__main__":
    main()
