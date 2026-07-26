import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "audio_normalizer_test_addon"


def load_module(name):
    package = sys.modules.get(PACKAGE)
    if package is None:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


class AudioCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_module("normalization.core")

    def test_default_is_study_speech_loudness(self):
        recipe = self.core.ProcessingRecipe()
        self.assertEqual(recipe.target_lufs, -18.0)
        self.assertIsNone(recipe.max_boost_db)

    def test_quiet_audio_uses_bounded_constant_gain(self):
        recipe = self.core.ProcessingRecipe(target_lufs=-20, max_boost_db=6)
        measurement = self.core.AudioMeasurement(-35, -20)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertEqual(decision.gain_db, 6)
        self.assertEqual(decision.outcome, "boost_limited")
        self.assertEqual(decision.expected_lufs, -29)

    def test_uncapped_boost_reaches_target_when_peak_safe(self):
        recipe = self.core.ProcessingRecipe(
            target_lufs=-18,
            true_peak_dbtp=-2,
            max_boost_db=None,
        )
        measurement = self.core.AudioMeasurement(-40, -30)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertEqual(decision.gain_db, 22)
        self.assertEqual(decision.outcome, "normalized")
        self.assertEqual(decision.expected_lufs, -18)

    def test_uncapped_boost_still_obeys_peak_ceiling(self):
        recipe = self.core.ProcessingRecipe(
            target_lufs=-18,
            true_peak_dbtp=-2,
            max_boost_db=None,
        )
        measurement = self.core.AudioMeasurement(-40, -10)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertEqual(decision.gain_db, 8)
        self.assertEqual(decision.outcome, "peak_limited")
        self.assertEqual(decision.expected_true_peak_dbtp, -2)

    def test_true_peak_ceiling_can_reduce_requested_gain(self):
        recipe = self.core.ProcessingRecipe(target_lufs=-20, true_peak_dbtp=-2)
        measurement = self.core.AudioMeasurement(-22, -1)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertEqual(decision.gain_db, -1)
        self.assertEqual(decision.outcome, "peak_limited")
        self.assertEqual(decision.expected_true_peak_dbtp, -2)

    def test_loud_audio_is_attenuated_to_target(self):
        recipe = self.core.ProcessingRecipe(target_lufs=-20)
        measurement = self.core.AudioMeasurement(-11, -0.5)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertEqual(decision.gain_db, -9)
        self.assertEqual(decision.expected_lufs, -20)

    def test_peak_ceiling_is_enforced_even_when_loudness_is_on_target(self):
        recipe = self.core.ProcessingRecipe(target_lufs=-20, true_peak_dbtp=-2)
        measurement = self.core.AudioMeasurement(-20, -0.5)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=False)
        self.assertTrue(decision.process)
        self.assertEqual(decision.gain_db, -1.5)
        self.assertEqual(decision.outcome, "peak_limited")

    def test_unmeasurable_audio_is_never_modified(self):
        recipe = self.core.ProcessingRecipe()
        measurement = self.core.AudioMeasurement(None, -4)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=True)
        self.assertFalse(decision.process)
        self.assertEqual(decision.outcome, "unmeasurable")

    def test_loudness_statistics_quantify_consistency_and_target_distance(self):
        stats = self.core.loudness_statistics([-24, -20, -16], -20)
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.mean_lufs, -20)
        self.assertAlmostEqual(stats.standard_deviation_lu, 3.266, places=3)
        self.assertAlmostEqual(
            stats.mean_distance_from_target_lu,
            2.667,
            places=3,
        )

    def test_loudness_statistics_ignore_unusable_values(self):
        stats = self.core.loudness_statistics([None, float("nan"), -18], -18)
        self.assertEqual(stats.count, 1)
        self.assertEqual(stats.standard_deviation_lu, 0)

    def test_conversion_still_runs_when_loudness_is_already_correct(self):
        recipe = self.core.ProcessingRecipe()
        measurement = self.core.AudioMeasurement(-18.2, -5)
        decision = self.core.decide_gain(measurement, recipe, needs_conversion=True)
        self.assertTrue(decision.process)
        self.assertEqual(decision.gain_db, 0)
        self.assertEqual(decision.outcome, "convert_only")

    def test_processed_filename_is_deterministic_and_recognizable(self):
        recipe = self.core.ProcessingRecipe()
        first = self.core.output_filename("声.opus", recipe, "a" * 64)
        repeated = self.core.output_filename("声.opus", recipe, "a" * 64)
        changed = self.core.output_filename("声.opus", recipe, "b" * 64)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertTrue(
            self.core.is_processed_for(first.rsplit(".", 1)[0], recipe)
        )

    def test_processed_filename_stays_within_portable_utf8_limit(self):
        recipe = self.core.ProcessingRecipe(output_format="source")
        filename = self.core.output_filename("声" * 200 + ".opus", recipe, "a" * 64)
        self.assertLessEqual(len(filename.encode("utf-8")), 240)
        self.assertTrue(filename.endswith(".opus"))
        self.assertTrue(
            self.core.is_processed_for(filename.rsplit(".", 1)[0], recipe)
        )

    def test_legacy_voicevox_marker_is_recognized_during_migration(self):
        recipe = self.core.ProcessingRecipe()
        legacy = f"clip__vva_v1_{recipe.token}_{'a' * 12}"
        self.assertTrue(self.core.is_processed_for(legacy, recipe))
        self.assertEqual(self.core.strip_processing_marker(legacy), "clip")

    def test_keep_format_recipe_preserves_the_source_extension(self):
        recipe = self.core.ProcessingRecipe(output_format="source")
        filename = self.core.output_filename("声.opus", recipe, "a" * 64)
        self.assertTrue(filename.endswith(".opus"))
        self.assertNotEqual(recipe.token, self.core.ProcessingRecipe().token)


class FakeNote:
    def __init__(self, note_id, fields):
        self.id = note_id
        self.fields = dict(fields)

    def keys(self):
        return list(self.fields)

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value):
        self.fields[key] = value


class FakeCollection:
    def __init__(self, notes):
        self.notes = {note.id: note for note in notes}
        self.updated = []

    def get_note(self, note_id):
        return self.notes.get(note_id)

    def update_note(self, note):
        self.updated.append(note.id)


class AnkiMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media = load_module("normalization.media")

    def test_discovers_all_fields_and_deduplicates_references(self):
        note = FakeNote(
            1,
            {
                "Word": "[sound:word.opus]",
                "Sentence": "x [sound:sentence.mp3] [sound:sentence.mp3]",
                "Unsafe": "[sound:../outside.wav]",
            },
        )
        collection = FakeCollection([note])
        result = self.media.discover_media_references(collection, [1])
        self.assertEqual(result.filenames, ["sentence.mp3", "word.opus"])
        self.assertEqual(result.unsafe_references, 1)

    def test_replacement_preserves_other_field_content(self):
        note = FakeNote(
            1,
            {"Audio": "before [sound:old.opus] after", "Text": "keep"},
        )
        collection = FakeCollection([note])
        discovery = self.media.discover_media_references(collection, [1])
        changed = self.media.replace_media_references(
            collection, discovery.references, {"old.opus": "new.mp3"}
        )
        self.assertEqual(changed, 1)
        self.assertEqual(note["Audio"], "before [sound:new.mp3] after")
        self.assertEqual(note["Text"], "keep")

    def test_replacement_updates_opt_in_filename_label(self):
        note = FakeNote(
            1,
            {
                "Audio": "[sound:old & loud.opus]",
                "Original filename": "old & loud.opus",
                "Current filename": (
                    '<span class="filename" data-audio-normalizer-filename>'
                    "old &amp; loud.opus</span>"
                ),
            },
        )
        collection = FakeCollection([note])
        discovery = self.media.discover_media_references(collection, [1])
        changed = self.media.replace_media_references(
            collection,
            discovery.references,
            {"old & loud.opus": "new & normalized.mp3"},
        )
        self.assertEqual(changed, 1)
        self.assertEqual(note["Audio"], "[sound:new & normalized.mp3]")
        self.assertEqual(
            note["Current filename"],
            '<span class="filename" data-audio-normalizer-filename>'
            "new &amp; normalized.mp3</span>",
        )
        self.assertEqual(note["Original filename"], "old & loud.opus")

    def test_replacement_uses_custom_undo_label(self):
        class UndoCollection(FakeCollection):
            def __init__(self, notes):
                super().__init__(notes)
                self.undo_labels = []
                self.merged = []

            def add_custom_undo_entry(self, label):
                self.undo_labels.append(label)
                return 42

            def merge_undo_entries(self, entry):
                self.merged.append(entry)

        note = FakeNote(1, {"Audio": "[sound:normalized.mp3]"})
        collection = UndoCollection([note])
        discovery = self.media.discover_media_references(collection, [1])
        self.media.replace_media_references(
            collection,
            discovery.references,
            {"normalized.mp3": "original.opus"},
            undo_label="Revert normalized card audio",
        )
        self.assertEqual(
            collection.undo_labels,
            ["Revert normalized card audio"],
        )
        self.assertEqual(collection.merged, [42])


class FakeBackend:
    def __init__(self, core, input_measurement=None):
        self.core = core
        self.transform_calls = []
        self.analyze_calls = 0
        self.input_measurement = input_measurement
        self.output_measurements = {}

    def analyze(self, path, _recipe):
        self.analyze_calls += 1
        if os.path.basename(path) in self.output_measurements:
            return self.output_measurements[os.path.basename(path)]
        if "__anorm_" in os.path.basename(path):
            return self.core.AudioMeasurement(-20, -3)
        return self.input_measurement or self.core.AudioMeasurement(-30, -12)

    def transform(
        self,
        input_path,
        output_path,
        gain_db,
        _recipe,
        output_format=None,
    ):
        self.transform_calls.append(
            (input_path, output_path, gain_db, output_format)
        )
        source_measurement = self.input_measurement or self.core.AudioMeasurement(
            -30, -12
        )
        self.output_measurements[os.path.basename(output_path)] = (
            self.core.AudioMeasurement(
                source_measurement.integrated_lufs + gain_db,
                source_measurement.true_peak_dbtp + gain_db,
            )
        )
        with open(input_path, "rb") as source, open(output_path, "wb") as output:
            output.write(b"mp3-" + source.read())


class FFmpegBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_module = load_module("normalization.ffmpeg_backend")
        cls.core = load_module("normalization.core")

    def test_short_audio_is_repeated_for_measurement_only(self):
        class Completed:
            returncode = 0

            def __init__(self, integrated):
                self.stderr = (
                    "{"
                    f'"input_i":"{integrated}",'
                    '"input_tp":"-8.0",'
                    '"target_offset":"0.0"'
                    "}"
                ).encode()

        backend = self.backend_module.FFmpegBackend("ffmpeg")
        commands = []
        responses = iter((Completed("-inf"), Completed("-20.0")))

        def fake_run(command):
            commands.append(command)
            return next(responses)

        backend._run = fake_run
        measurement = backend.analyze("word.wav", self.core.ProcessingRecipe())
        self.assertEqual(measurement.integrated_lufs, -20.0)
        self.assertEqual(len(commands), 2)
        self.assertNotIn("-stream_loop", commands[0])
        self.assertIn("-stream_loop", commands[1])


class FFmpegProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider_module = load_module("ffmpeg")

    def test_missing_ffmpeg_is_not_downloaded(self):
        provider = self.provider_module.FFmpegProvider()
        provider.discover = lambda: None
        self.assertIsNone(provider.ensure_available())
        self.assertIn("does not download", provider.last_error)

    def test_version_is_read_from_ffmpeg_output(self):
        class Completed:
            returncode = 0
            stdout = b"ffmpeg version 8.1.2 Copyright (c) FFmpeg developers\n"
            stderr = b""

        provider = self.provider_module.FFmpegProvider()
        provider_type = self.provider_module.FFmpegProvider
        real_probe = provider_type.__dict__["_probe"]
        provider_type._probe = staticmethod(lambda *_args, **_kwargs: Completed())
        try:
            self.assertEqual(provider._version("ffmpeg"), "8.1.2")
        finally:
            provider_type._probe = real_probe

    def test_ffmpeg_without_required_encoder_is_rejected(self):
        class Completed:
            returncode = 0
            stderr = b""

            def __init__(self, stdout):
                self.stdout = stdout

        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, "ffmpeg")
            Path(executable).write_bytes(b"placeholder")
            real_run = self.provider_module.subprocess.run

            def fake_run(command, **_kwargs):
                if "-filters" in command:
                    return Completed(b" loudnorm ")
                return Completed(b" libmp3lame aac flac pcm_s16le ")

            self.provider_module.subprocess.run = fake_run
            try:
                self.assertFalse(
                    self.provider_module.FFmpegProvider._is_usable(executable)
                )
            finally:
                self.provider_module.subprocess.run = real_run

    def test_macos_homebrew_ffmpeg_is_found_without_shell_path(self):
        provider = self.provider_module.FFmpegProvider()
        real_platform = self.provider_module.sys.platform
        real_which = self.provider_module.shutil.which
        real_is_usable = provider._is_usable
        self.provider_module.sys.platform = "darwin"
        self.provider_module.shutil.which = lambda _name: None
        provider._is_usable = lambda path: path == "/opt/homebrew/bin/ffmpeg"
        try:
            self.assertEqual(
                provider.discover(), os.path.realpath("/opt/homebrew/bin/ffmpeg")
            )
        finally:
            self.provider_module.sys.platform = real_platform
            self.provider_module.shutil.which = real_which
            provider._is_usable = real_is_usable

    def test_explicit_override_is_checked_first(self):
        provider = self.provider_module.FFmpegProvider()
        variable = self.provider_module.FFMPEG_OVERRIDE_ENV
        previous = os.environ.get(variable)
        os.environ[variable] = "/custom/ffmpeg"
        real_which = self.provider_module.shutil.which
        self.provider_module.shutil.which = lambda _name: "/path/ffmpeg"
        try:
            self.assertEqual(provider._system_candidates()[0], "/custom/ffmpeg")
        finally:
            self.provider_module.shutil.which = real_which
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


class AddonBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("tools.build_addon")

    def test_build_is_reproducible_and_contains_third_party_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.ankiaddon"
            second = Path(directory) / "second.ankiaddon"
            self.builder.build(first)
            self.builder.build(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = set(archive.namelist())
            self.assertIn("THIRD_PARTY_NOTICES.md", names)
            self.assertNotIn("CHANGELOG.md", names)
            self.assertNotIn("RELEASE.md", names)
            self.assertNotIn("ffmpeg_downloads.json", names)


class AudioServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_module("normalization.core")
        cls.service_module = load_module("normalization.service")
        cls.state_module = load_module("normalization.state")

    def test_processes_once_and_reuses_deterministic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "clip.opus")
            with open(source, "wb") as output:
                output.write(b"audio")
            backend = FakeBackend(self.core)
            service = self.service_module.AudioNormalizationService(
                backend, directory, self.core.ProcessingRecipe()
            )
            first = service.process(["clip.opus"])
            second = service.process(["clip.opus"])
            self.assertEqual(first.changed, 1)
            self.assertEqual(first.normalized, 1)
            self.assertEqual(first.converted, 1)
            self.assertEqual(second.files[0].status, "reused")
            self.assertEqual(second.changed, 0)
            self.assertEqual(second.normalized, 0)
            self.assertEqual(len(backend.transform_calls), 1)
            self.assertTrue(os.path.isfile(os.path.join(
                directory, first.files[0].output_filename
            )))

    def test_batch_reports_measured_before_and_after_effectiveness(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "clip.opus")
            Path(source).write_bytes(b"audio")
            service = self.service_module.AudioNormalizationService(
                FakeBackend(
                    self.core,
                    self.core.AudioMeasurement(-30, -20),
                ),
                directory,
                self.core.ProcessingRecipe(),
            )
            batch = service.process(["clip.opus"])
            before, after = batch.effectiveness
            self.assertEqual(before.mean_lufs, -30)
            self.assertEqual(before.mean_distance_from_target_lu, 12)
            self.assertEqual(after.mean_lufs, -18)
            self.assertEqual(after.mean_distance_from_target_lu, 0)
            self.assertEqual(
                batch.files[0].output_measurement.integrated_lufs,
                -18,
            )

    def test_batch_reports_progress_after_each_unique_file(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("b.opus", "a.opus"):
                with open(os.path.join(directory, filename), "wb") as output:
                    output.write(b"audio")
            service = self.service_module.AudioNormalizationService(
                FakeBackend(self.core),
                directory,
                self.core.ProcessingRecipe(),
            )
            updates = []
            service.process(
                ["b.opus", "a.opus", "a.opus"],
                progress_callback=lambda completed, total, filename, item: updates.append(
                    (completed, total, filename, item.status if item else None)
                ),
            )
            self.assertEqual(
                updates,
                [
                    (0, 2, "a.opus", None),
                    (1, 2, "a.opus", "peak_limited"),
                    (1, 2, "b.opus", None),
                    (2, 2, "b.opus", "peak_limited"),
                ],
            )

    def test_state_cache_avoids_remeasuring_unchanged_ready_mp3(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "ready.mp3")
            state_path = os.path.join(directory, "state.json")
            with open(source, "wb") as output:
                output.write(b"audio")
            backend = FakeBackend(
                self.core, self.core.AudioMeasurement(-18.1, -4)
            )
            state = self.state_module.ProcessingState(state_path)
            service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(),
                state=state,
            )
            first = service.process(["ready.mp3"])
            restarted_service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(),
                state=self.state_module.ProcessingState(state_path),
            )
            real_sha256_file = self.service_module.sha256_file
            hash_calls = []

            def tracked_sha256_file(path):
                hash_calls.append(path)
                return real_sha256_file(path)

            self.service_module.sha256_file = tracked_sha256_file
            try:
                second = restarted_service.process(["ready.mp3"])
            finally:
                self.service_module.sha256_file = real_sha256_file
            self.assertEqual(first.files[0].status, "already_ready")
            self.assertEqual(second.files[0].status, "already_ready_cached")
            self.assertEqual(backend.analyze_calls, 1)
            self.assertEqual(hash_calls, [])

    def test_metadata_change_falls_back_to_hash_and_remeasurement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "ready.mp3")
            with open(source, "wb") as output:
                output.write(b"first")
            backend = FakeBackend(
                self.core, self.core.AudioMeasurement(-18.1, -4)
            )
            state = self.state_module.ProcessingState(
                os.path.join(directory, "state.json")
            )
            service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(),
                state=state,
            )
            service.process(["ready.mp3"])
            old_mtime = os.stat(source).st_mtime_ns
            with open(source, "wb") as output:
                output.write(b"other")
            os.utime(source, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
            second = service.process(["ready.mp3"])
            self.assertEqual(second.files[0].status, "already_ready")
            self.assertEqual(backend.analyze_calls, 2)

    def test_processed_output_uses_metadata_fast_path_on_later_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.opus")
            state_path = os.path.join(directory, "state.json")
            with open(source, "wb") as output:
                output.write(b"audio")
            state = self.state_module.ProcessingState(
                state_path
            )
            service = self.service_module.AudioNormalizationService(
                FakeBackend(self.core),
                directory,
                self.core.ProcessingRecipe(),
                state=state,
            )
            normalized = service.process(["source.opus"]).files[0]
            restarted_service = self.service_module.AudioNormalizationService(
                FakeBackend(self.core),
                directory,
                self.core.ProcessingRecipe(),
                state=self.state_module.ProcessingState(state_path),
            )
            real_sha256_file = self.service_module.sha256_file
            hash_calls = []

            def tracked_sha256_file(path):
                hash_calls.append(path)
                return real_sha256_file(path)

            self.service_module.sha256_file = tracked_sha256_file
            try:
                checked = restarted_service.process(
                    [normalized.output_filename]
                ).files[0]
            finally:
                self.service_module.sha256_file = real_sha256_file
            self.assertEqual(checked.status, "already_processed")
            self.assertEqual(hash_calls, [])

    def test_batch_can_stop_cleanly_between_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("a.mp3", "b.mp3"):
                with open(os.path.join(directory, filename), "wb") as output:
                    output.write(b"audio")
            service = self.service_module.AudioNormalizationService(
                FakeBackend(
                    self.core, self.core.AudioMeasurement(-20.1, -4)
                ),
                directory,
                self.core.ProcessingRecipe(),
            )
            checks = 0

            def should_cancel():
                nonlocal checks
                checks += 1
                return checks > 1

            batch = service.process(
                ["a.mp3", "b.mp3"],
                should_cancel=should_cancel,
            )
            self.assertTrue(batch.cancelled)
            self.assertEqual(
                [item.source_filename for item in batch.files],
                ["a.mp3"],
            )

    def test_missing_processed_file_is_reported(self):
        recipe = self.core.ProcessingRecipe()
        filename = self.core.output_filename("gone.opus", recipe, "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            service = self.service_module.AudioNormalizationService(
                FakeBackend(self.core), directory, recipe
            )
            item = service.process([filename]).files[0]
            self.assertEqual(item.status, "missing")

    def test_failed_output_verification_leaves_source_and_removes_new_copy(self):
        class UnsafeOutputBackend(FakeBackend):
            def analyze(self, path, _recipe):
                if "__anorm_" in os.path.basename(path):
                    return self.core.AudioMeasurement(-10, 0)
                return self.core.AudioMeasurement(-30, -12)

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.opus")
            with open(source, "wb") as output:
                output.write(b"original")
            service = self.service_module.AudioNormalizationService(
                UnsafeOutputBackend(self.core),
                directory,
                self.core.ProcessingRecipe(),
            )
            item = service.process(["source.opus"]).files[0]
            self.assertEqual(item.status, "error")
            self.assertEqual(Path(source).read_bytes(), b"original")
            self.assertEqual(
                [name for name in os.listdir(directory) if "__anorm_" in name],
                [],
            )

    def test_unrelated_existing_output_is_not_reused(self):
        class WrongExistingOutputBackend(FakeBackend):
            def analyze(self, path, recipe):
                if "__anorm_" in os.path.basename(path):
                    return self.core.AudioMeasurement(-15, -3)
                return super().analyze(path, recipe)

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.opus")
            Path(source).write_bytes(b"original")
            recipe = self.core.ProcessingRecipe()
            source_hash = self.service_module.sha256_file(source)
            destination = self.core.output_filename(
                "source.opus", recipe, source_hash
            )
            Path(directory, destination).write_bytes(b"unrelated audio")
            service = self.service_module.AudioNormalizationService(
                WrongExistingOutputBackend(self.core), directory, recipe
            )
            item = service.process(["source.opus"]).files[0]
            self.assertEqual(item.status, "error")
            self.assertIn("unexpected loudness", item.detail)

    def test_recipe_change_reuses_original_instead_of_transcoding_mp3_again(self):
        with tempfile.TemporaryDirectory() as directory:
            original = os.path.join(directory, "clip.opus")
            with open(original, "wb") as output:
                output.write(b"original")
            state = self.state_module.ProcessingState(
                os.path.join(directory, "state.json")
            )
            backend = FakeBackend(self.core)
            first_service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(target_lufs=-20),
                state=state,
            )
            first = first_service.process(["clip.opus"]).files[0]
            second_service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(target_lufs=-18),
                state=state,
            )
            second = second_service.process([first.output_filename]).files[0]
            self.assertIsNotNone(second.output_filename)
            self.assertEqual(backend.transform_calls[-1][0], original)
            self.assertEqual(
                state.origin_for(second.output_filename),
                "clip.opus",
            )

    def test_recipe_change_rejects_unsafe_cached_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = self.core.ProcessingRecipe(target_lufs=-18)
            old_recipe = self.core.ProcessingRecipe(target_lufs=-20)
            processed_name = self.core.output_filename(
                "clip.opus", old_recipe, "a" * 64
            )
            processed_path = os.path.join(directory, processed_name)
            with open(processed_path, "wb") as output:
                output.write(b"processed")
            state = self.state_module.ProcessingState(
                os.path.join(directory, "state.json")
            )
            state.remember(
                processed_name,
                "irrelevant",
                old_recipe.token,
                "verified",
                origin="../outside.opus",
            )
            service = self.service_module.AudioNormalizationService(
                FakeBackend(self.core), directory, recipe, state=state
            )
            item = service.process([processed_name]).files[0]
            self.assertEqual(item.status, "error")
            self.assertIn("original could not be identified", item.detail)

    def test_keep_format_normalizes_opus_without_converting_to_mp3(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "clip.opus")
            with open(source, "wb") as output:
                output.write(b"audio")
            backend = FakeBackend(self.core)
            service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(output_format="source"),
            )
            item = service.process(["clip.opus"]).files[0]
            self.assertTrue(item.output_filename.endswith(".opus"))
            self.assertEqual(backend.transform_calls[-1][3], "opus")

    def test_keep_format_leaves_in_tolerance_opus_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "ready.opus")
            with open(source, "wb") as output:
                output.write(b"original")
            backend = FakeBackend(
                self.core, self.core.AudioMeasurement(-18.1, -4)
            )
            service = self.service_module.AudioNormalizationService(
                backend,
                directory,
                self.core.ProcessingRecipe(output_format="source"),
            )
            item = service.process(["ready.opus"]).files[0]
            self.assertEqual(item.status, "already_ready")
            self.assertEqual(backend.transform_calls, [])
            self.assertEqual(Path(source).read_bytes(), b"original")

class SearchScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = load_module("normalization.scope")

    def test_deck_and_additional_filters_form_one_bounded_search(self):
        self.assertEqual(
            self.scope.effective_search("Japanese::Listening", "tag:new OR tag:hard"),
            'deck:"Japanese::Listening" (tag:new OR tag:hard)',
        )

    def test_deck_names_are_escaped(self):
        self.assertEqual(
            self.scope.deck_search('Deck "One"'),
            r'deck:"Deck \"One\""',
        )

    def test_legacy_deck_query_migrates_to_first_class_scope(self):
        self.assertEqual(
            self.scope.split_legacy_scope(
                "", 'deck:"Japanese Listening" note:"Sentence"'
            ),
            ("Japanese Listening", 'note:"Sentence"'),
        )
        self.assertEqual(
            self.scope.split_legacy_scope("", "deck:Kikitori"),
            ("Kikitori", ""),
        )

    def test_current_anki_deck_api_is_sorted_and_deduplicated(self):
        manager = types.SimpleNamespace(
            all_names_and_ids=lambda: [
                types.SimpleNamespace(name="Zulu"),
                types.SimpleNamespace(name="alpha"),
                types.SimpleNamespace(name="Zulu"),
            ]
        )
        self.assertEqual(
            self.scope.available_deck_names(manager),
            ["alpha", "Zulu"],
        )

    def test_browser_selection_is_intersected_with_current_search(self):
        self.assertEqual(
            self.scope.browser_scope_search(
                'deck:"Japanese" tag:listening',
                [42, 7, 42],
            ),
            '(deck:"Japanese" tag:listening) (nid:7,42)',
        )

    def test_browser_search_is_preserved_without_a_selection(self):
        self.assertEqual(
            self.scope.browser_scope_search("is:due\n-tag:suspend", []),
            "is:due -tag:suspend",
        )

    def test_browser_selection_rejects_invalid_note_ids(self):
        with self.assertRaises(ValueError):
            self.scope.browser_scope_search("", ["not-an-id"])


class ProcessingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_module = load_module("normalization.state")

    def test_malformed_entries_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory, "state.json")
            state_path.write_text(
                '{"version": 1, "entries": ["not", "a", "mapping"]}',
                encoding="utf-8",
            )
            state = self.state_module.ProcessingState(str(state_path))
            self.assertEqual(state.entries, {})
            self.assertIsNone(state.origin_for("anything.mp3"))


class RevertPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_module("normalization.core")
        cls.recovery = load_module("normalization.recovery")
        cls.state_module = load_module("normalization.state")

    def test_uses_recorded_original_for_generated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            original = "voice.opus"
            Path(directory, original).write_bytes(b"original")
            recipe = self.core.ProcessingRecipe()
            generated = self.core.output_filename(
                original, recipe, "a" * 64
            )
            Path(directory, generated).write_bytes(b"normalized")
            state = self.state_module.ProcessingState(
                os.path.join(directory, "state.json")
            )
            state.remember(
                generated,
                "b" * 64,
                recipe.token,
                "verified",
                origin=original,
                source_path=os.path.join(directory, generated),
            )
            plan = self.recovery.plan_revert(
                directory, [generated], state
            )
            self.assertEqual(plan.replacements, {generated: original})
            self.assertEqual(plan.unresolved, {})

    def test_infers_one_original_when_state_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            original = "voice.wav"
            Path(directory, original).write_bytes(b"original")
            generated = self.core.output_filename(
                original,
                self.core.ProcessingRecipe(),
                "a" * 64,
            )
            Path(directory, generated).write_bytes(b"normalized")
            plan = self.recovery.plan_revert(
                directory, [generated], state=None
            )
            self.assertEqual(plan.replacements, {generated: original})

    def test_refuses_ambiguous_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "voice.wav").write_bytes(b"wav")
            Path(directory, "voice.opus").write_bytes(b"opus")
            generated = self.core.output_filename(
                "voice.wav",
                self.core.ProcessingRecipe(),
                "a" * 64,
            )
            Path(directory, generated).write_bytes(b"normalized")
            plan = self.recovery.plan_revert(
                directory, [generated], state=None
            )
            self.assertEqual(plan.replacements, {})
            self.assertIn("Multiple", plan.unresolved[generated])

    def test_does_not_guess_when_recorded_original_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "voice.wav").write_bytes(b"unrelated")
            recipe = self.core.ProcessingRecipe()
            generated = self.core.output_filename(
                "voice.opus",
                recipe,
                "a" * 64,
            )
            Path(directory, generated).write_bytes(b"normalized")
            state = self.state_module.ProcessingState(
                os.path.join(directory, "state.json")
            )
            state.remember(
                generated,
                "b" * 64,
                recipe.token,
                "verified",
                origin="voice.opus",
            )
            plan = self.recovery.plan_revert(
                directory, [generated], state
            )
            self.assertEqual(plan.replacements, {})
            self.assertIn("voice.opus", plan.unresolved[generated])

    def test_leaves_original_references_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "voice.mp3").write_bytes(b"audio")
            plan = self.recovery.plan_revert(
                directory, ["voice.mp3"], state=None
            )
            self.assertEqual(plan.already_original, 1)
            self.assertEqual(plan.replacements, {})


if __name__ == "__main__":
    unittest.main()
