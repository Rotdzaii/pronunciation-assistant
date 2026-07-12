from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app.alignment.audio_preparation import PreparedMfaAudio, prepare_audio_for_mfa
from app.alignment.alignment_service import align_audio
from app.scorers import cnn_attention_scorer
from app.alignment.mfa_aligner import (
    _mfa_command_context,
    _mfa_command_prefix,
    _write_debug_artifacts,
    build_mfa_process_diagnostic,
    run_mfa_alignment,
    run_mfa_preflight,
)
from app.alignment.quality import validate_alignment_quality
from app.alignment.textgrid_parser import parse_textgrid
from app.alignment.transcript import normalize_and_check_transcript
from app.contracts.alignment_contract import AlignmentError
from app.contracts.webhook_payload import build_failed_webhook_payload, build_success_webhook_payload
from app.scorers.cnn_attention_scorer import CNNAttentionScorerError, score_aligned_audio_context


FIXTURES = Path(__file__).parent / "fixtures"


def write_tone(path: Path, seconds: float = 1.0) -> None:
    frame_count = int(16000 * seconds)
    samples = bytearray()
    for index in range(frame_count):
        value = 8000 if (index // 40) % 2 else -8000
        samples.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(bytes(samples))


def prepare_test_audio(source: Path, destination: Path) -> PreparedMfaAudio:
    shutil.copy2(source, destination)
    return PreparedMfaAudio(destination, 1.0, 16000, 16000, 0.2, 0.1)


class MfaAlignmentUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_environment = patch.dict(
            os.environ,
            {"MFA_RUNTIME": "conda", "MFA_CONDA_ENV": "aligner"},
            clear=False,
        )
        self.runtime_environment.start()

    def tearDown(self) -> None:
        self.runtime_environment.stop()

    def test_conda_runtime_uses_conda_run_and_ignores_wsl_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MFA_RUNTIME": "conda",
                "MFA_CONDA_ENV": "aligner",
                "MFA_WSL_DISTRO": "Ubuntu",
                "MFA_WSL_USER": "phoenix",
                "MFA_WSL_BINARY": "/home/phoenix/run_mfa.sh",
            },
            clear=False,
        ), patch(
            "app.alignment.mfa_aligner._mfa_command_prefix",
            return_value=["conda", "run", "-n", "aligner", "mfa"],
        ):
            use_wsl, command, context = _mfa_command_context()
        self.assertFalse(use_wsl)
        self.assertEqual(command, ["conda", "run", "-n", "aligner", "mfa"])
        self.assertEqual(context["mfa_runtime"], "conda")
        self.assertEqual(context["command_mode"], "conda_run")

    def test_wsl_runtime_requires_explicit_selection_and_builds_wsl_command(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MFA_RUNTIME": "wsl",
                "MFA_WSL_DISTRO": "Ubuntu",
                "MFA_WSL_USER": "phoenix",
                "MFA_WSL_BINARY": "/home/phoenix/run_mfa.sh",
            },
            clear=False,
        ):
            use_wsl, command, context = _mfa_command_context()
        self.assertTrue(use_wsl)
        self.assertEqual(command, ["wsl", "-d", "Ubuntu", "-u", "phoenix", "--", "/home/phoenix/run_mfa.sh"])
        self.assertEqual(context["mfa_runtime"], "wsl")

    def test_wsl_runtime_with_missing_configuration_fails_without_conda_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"MFA_RUNTIME": "wsl", "MFA_WSL_DISTRO": "", "MFA_WSL_USER": "", "MFA_WSL_BINARY": ""},
            clear=False,
        ):
            with self.assertRaises(AlignmentError) as context:
                _mfa_command_context()
        self.assertEqual(context.exception.code, "mfa_wsl_configuration_invalid")

    def test_invalid_runtime_is_rejected_without_fallback(self) -> None:
        with patch.dict(os.environ, {"MFA_RUNTIME": "docker"}, clear=False):
            with self.assertRaises(AlignmentError) as context:
                _mfa_command_context()
        self.assertEqual(context.exception.code, "mfa_runtime_invalid")

    def test_conda_runtime_does_not_fallback_to_wsl_when_conda_env_is_missing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MFA_RUNTIME": "conda",
                "MFA_CONDA_ENV": "",
                "MFA_WSL_DISTRO": "Ubuntu",
                "MFA_WSL_USER": "phoenix",
                "MFA_WSL_BINARY": "/home/phoenix/run_mfa.sh",
            },
            clear=False,
        ):
            with self.assertRaises(AlignmentError) as context:
                _mfa_command_context()
        self.assertEqual(context.exception.code, "mfa_conda_configuration_invalid")
    def test_valid_audio_and_transcript_are_normalized_for_mfa(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            prepared = Path(temp) / "prepared.wav"
            write_tone(source)
            result = prepare_audio_for_mfa(source, prepared)
            self.assertEqual(result.sample_rate, 16000)
            self.assertGreater(result.duration_seconds, 0.9)
            with wave.open(str(prepared)) as wav_file:
                self.assertEqual((wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()), (16000, 1, 2))
            transcript = normalize_and_check_transcript("  Cat,  cat! ")
            self.assertEqual(transcript.text, "Cat cat")

    def test_empty_transcript_is_rejected(self) -> None:
        with self.assertRaisesRegex(AlignmentError, "empty") as context:
            normalize_and_check_transcript("  ...  ")
        self.assertEqual(context.exception.code, "empty_transcript")

    def test_empty_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "empty.wav"
            source.touch()
            with self.assertRaises(AlignmentError) as context:
                prepare_audio_for_mfa(source, Path(temp) / "prepared.wav")
        self.assertEqual(context.exception.code, "audio_empty")

    def test_corrupt_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "corrupt.wav"
            source.write_bytes(b"not a wav file")
            with self.assertRaises(AlignmentError) as context:
                prepare_audio_for_mfa(source, Path(temp) / "prepared.wav")
        self.assertEqual(context.exception.code, "audio_invalid")

    def test_oov_is_reported_from_plain_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dictionary = Path(temp) / "dictionary.txt"
            dictionary.write_text("cat K AE T\n", encoding="utf-8")
            transcript = normalize_and_check_transcript("cat zzz", dictionary)
        self.assertEqual(transcript.oov_words, ["zzz"])
        self.assertTrue(transcript.dictionary_checked)

    def test_missing_mfa_executable_is_classified(self) -> None:
        with patch("app.alignment.mfa_aligner.shutil.which", return_value=None):
            with self.assertRaises(AlignmentError) as context:
                _mfa_command_prefix("missing-mfa", None)
        self.assertEqual(context.exception.code, "mfa_not_installed")

    def test_mfa_timeout_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "input.wav"
            dictionary = Path(temp) / "dictionary.txt"
            write_tone(audio)
            dictionary.write_text("cat K AE T\n", encoding="utf-8")
            prepared = PreparedMfaAudio(Path(temp) / "prepared.wav", 1.0, 16000, 16000, 0.2, 0.1)
            with patch("app.alignment.mfa_aligner._mfa_command_prefix", return_value=["mfa"]), patch(
                "app.alignment.mfa_aligner.prepare_audio_for_mfa", return_value=prepared
            ), patch("app.alignment.mfa_aligner.subprocess.run", side_effect=subprocess.TimeoutExpired(["mfa"], 1)):
                with self.assertRaises(AlignmentError) as context:
                    run_mfa_alignment(audio, "cat", dictionary_path=dictionary, acoustic_model_path="english_mfa")
        self.assertEqual(context.exception.code, "mfa_timeout")

    def test_mfa_success_without_textgrid_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "input.wav"
            dictionary = Path(temp) / "dictionary.txt"
            write_tone(audio)
            dictionary.write_text("cat K AE T\n", encoding="utf-8")
            prepared = PreparedMfaAudio(Path(temp) / "prepared.wav", 1.0, 16000, 16000, 0.2, 0.1)
            completed = subprocess.CompletedProcess(["mfa"], 0, "", "")
            with patch("app.alignment.mfa_aligner._mfa_command_prefix", return_value=["mfa"]), patch(
                "app.alignment.mfa_aligner.prepare_audio_for_mfa", return_value=prepared
            ), patch("app.alignment.mfa_aligner.subprocess.run", return_value=completed):
                with self.assertRaises(AlignmentError) as context:
                    run_mfa_alignment(audio, "cat", dictionary_path=dictionary, acoustic_model_path="english_mfa")
        self.assertEqual(context.exception.code, "textgrid_missing")

    def test_mfa_returncode_zero_parses_generated_textgrid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "input.wav"
            dictionary = root / "dictionary.txt"
            write_tone(audio)
            dictionary.write_text("cat K AE T\n", encoding="utf-8")

            def complete_alignment(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                output_dir = Path(command[5])
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(FIXTURES / "valid.TextGrid", output_dir / "input.TextGrid")
                return subprocess.CompletedProcess(command, 0, "MFA completed", "")

            with patch("app.alignment.mfa_aligner._mfa_command_prefix", return_value=["mfa"]), patch(
                "app.alignment.mfa_aligner.prepare_audio_for_mfa", side_effect=prepare_test_audio
            ), patch("app.alignment.mfa_aligner.subprocess.run", side_effect=complete_alignment):
                result = run_mfa_alignment(audio, "cat", dictionary_path=dictionary, acoustic_model_path="english_mfa")
        self.assertEqual(result["alignment_status"], "success")
        self.assertEqual(len(result["phones"]), 3)
        self.assertEqual(result["metadata"]["mfa_runtime"], "conda")
        self.assertEqual(result["metadata"]["mfa_command_mode"], "conda_run")
        self.assertEqual(result["metadata"]["phone_span_fill_ratio"], 1.0)
        self.assertNotIn("speech_relative_coverage_ratio", result["metadata"])

    def test_nonzero_diagnostic_keeps_mfa_error_and_redacts_paths_and_urls(self) -> None:
        diagnostic = build_mfa_process_diagnostic(
            stage="align",
            command_args=["conda", "run", "-n", "aligner", "mfa", "align", r"C:\\temp\\corpus", "english_us_mfa", "english_mfa", r"C:\\temp\\out"],
            command_context={"command_mode": "conda_run", "conda_env": "aligner"},
            return_code=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                r"ValueError: KaldiError at C:\\Users\\Admin\\AppData\\Local\\Temp\\mfa\\log.txt "
                "https://example.test/file?token=secret-value"
            ),
            dictionary="english_us_mfa",
            acoustic_model="english_mfa",
            corpus_dir=Path(r"C:\\temp\\corpus"),
            output_dir=Path(r"C:\\temp\\out"),
        )
        rendered = str(diagnostic)
        self.assertIn("ValueError: KaldiError", rendered)
        self.assertIn("Traceback", diagnostic["exception_or_traceback_tail"])
        self.assertNotIn("C:\\Users", rendered)
        self.assertNotIn("secret-value", rendered)
        self.assertNotIn("https://example.test", rendered)
        self.assertEqual(diagnostic["command_mode"], "conda_run")

    def test_nonzero_diagnostic_uses_stdout_when_stderr_is_empty(self) -> None:
        diagnostic = build_mfa_process_diagnostic(
            stage="align",
            command_args=["mfa", "align"],
            command_context={"command_mode": "mfa_direct", "conda_env": None},
            return_code=1,
            stdout="MFAException: dictionary mismatch",
            stderr="",
        )
        self.assertEqual(diagnostic["primary_output_tail"], "MFAException: dictionary mismatch")

    def test_wsl_corpus_and_output_paths_are_redacted_from_diagnostic(self) -> None:
        diagnostic = build_mfa_process_diagnostic(
            stage="align",
            command_args=["wsl", "--", "mfa", "align", "/mnt/c/Users/Admin/AppData/Local/Temp/corpus", "english_us_mfa", "english_mfa", "/mnt/c/Users/Admin/AppData/Local/Temp/aligned"],
            command_context={"command_mode": "wsl", "conda_env": None},
            return_code=2,
            stdout="",
            stderr="Model path /mnt/c/Users/Admin/AppData/Local/Temp/models is unavailable.",
            dictionary="english_us_mfa",
            acoustic_model="english_mfa",
            corpus_dir=Path(r"C:\\Users\\Admin\\AppData\\Local\\Temp\\corpus"),
            output_dir=Path(r"C:\\Users\\Admin\\AppData\\Local\\Temp\\aligned"),
        )
        rendered = str(diagnostic)
        self.assertNotIn("/mnt/c/Users", rendered)
        self.assertIn("<corpus-dir>", diagnostic["command"])
        self.assertIn("<output-dir>", diagnostic["command"])

    def test_debug_artifacts_are_only_created_for_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared.wav"
            transcript = root / "transcript.lab"
            write_tone(prepared)
            transcript.write_text("cat\n", encoding="utf-8")
            diagnostic = {"command": ["mfa", "align", "<corpus-dir>"], "stderr_tail": "KaldiError"}
            self.assertFalse(
                _write_debug_artifacts(
                    None,
                    job_id="job-1",
                    prepared_audio=prepared,
                    transcript_lab=transcript,
                    stdout="",
                    stderr="C:\\temp\\bad",
                    diagnostic=diagnostic,
                )
            )
            debug_root = root / "debug"
            self.assertTrue(
                _write_debug_artifacts(
                    debug_root,
                    job_id="job-1",
                    prepared_audio=prepared,
                    transcript_lab=transcript,
                    stdout="",
                    stderr="C:\\temp\\bad",
                    diagnostic=diagnostic,
                )
            )
            artifact_dir = next(debug_root.iterdir())
            self.assertTrue((artifact_dir / "prepared.wav").is_file())
            self.assertTrue((artifact_dir / "transcript.lab").is_file())
            self.assertNotIn("C:\\temp", (artifact_dir / "stderr.txt").read_text(encoding="utf-8"))

    def test_debug_runner_cleans_temp_work_directory_without_debug_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "input.wav"
            dictionary = root / "dictionary.txt"
            mfa_temp = root / "mfa-temp"
            write_tone(audio)
            dictionary.write_text("cat K AE T\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(["mfa"], 1, "", "KaldiError")
            with patch.dict(
                os.environ,
                {"MFA_DEBUG": "true", "MFA_DEBUG_DIR": "", "MFA_TEMP_DIR": str(mfa_temp)},
                clear=False,
            ), patch("app.alignment.mfa_aligner._mfa_command_prefix", return_value=["mfa"]), patch(
                "app.alignment.mfa_aligner.prepare_audio_for_mfa", side_effect=prepare_test_audio
            ), patch("app.alignment.mfa_aligner.subprocess.run", return_value=completed):
                with self.assertRaises(AlignmentError) as context:
                    run_mfa_alignment(audio, "cat", dictionary_path=dictionary, acoustic_model_path="english_mfa")
        self.assertEqual(context.exception.code, "mfa_nonzero_exit")
        self.assertEqual(list(mfa_temp.glob("mfa-align-*")), [])

    def test_mfa_preflight_lists_version_and_models(self) -> None:
        responses = [
            subprocess.CompletedProcess(["mfa", "version"], 0, "3.3.8", ""),
            subprocess.CompletedProcess(["mfa", "model", "list", "dictionary"], 0, "english_us_mfa\n", ""),
            subprocess.CompletedProcess(["mfa", "model", "list", "acoustic"], 0, "english_mfa\n", ""),
        ]
        with patch.dict(os.environ, {"MFA_CONDA_ENV": "aligner"}, clear=False), patch(
            "app.alignment.mfa_aligner._mfa_command_prefix", return_value=["conda", "run", "-n", "aligner", "mfa"]
        ), patch("app.alignment.mfa_aligner.subprocess.run", side_effect=responses):
            result = run_mfa_preflight("english_us_mfa", "english_mfa")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["dictionary_listed"])
        self.assertTrue(result["acoustic_model_listed"])

    def test_mfa_failure_returns_explicit_fallback_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "input.wav"
            write_tone(audio)
            with patch.dict(os.environ, {"ALIGNMENT_MODE": "mfa", "ALLOW_ALIGNMENT_FALLBACK": "true"}), patch(
                "app.alignment.alignment_service.run_mfa_alignment",
                side_effect=AlignmentError("MFA missing", code="mfa_not_installed"),
            ):
                result = align_audio(audio, "cat", canonical_phones=["K", "AE", "T"], job_id="job-1")
        self.assertEqual(result["alignment_status"], "fallback")
        self.assertEqual(result["alignment_source"], "fallback")
        self.assertEqual(result["metadata"]["mfa_error"]["code"], "mfa_not_installed")
        self.assertEqual(result["quality"]["status"], "warning")

    def test_public_alignment_contract_does_not_include_raw_mfa_stderr(self) -> None:
        raw_stderr = r"KaldiError C:\\Users\\Admin\\AppData\\Local\\Temp\\mfa.log token=secret-value"
        with patch.dict(os.environ, {"ALIGNMENT_MODE": "mfa", "ALLOW_ALIGNMENT_FALLBACK": "false"}), patch(
            "app.alignment.alignment_service.run_mfa_alignment",
            side_effect=AlignmentError("MFA returned exit code 1.", code="mfa_nonzero_exit", details={"stderr": raw_stderr}),
        ):
            result = align_audio("not-used.wav", "cat")
        self.assertEqual(result["error"]["code"], "mfa_nonzero_exit")
        self.assertNotIn("KaldiError", str(result))
        self.assertNotIn("secret-value", str(result))
        payload = build_failed_webhook_payload(
            "job-1",
            "Alignment failed.",
            {
                "status": "failed",
                "score": None,
                "problem_phonemes": [],
                "feedback": {},
                "metadata": result["metadata"],
            },
        )
        self.assertNotIn("KaldiError", str(payload))
        self.assertNotIn("secret-value", str(payload))

    def test_context_scorer_does_not_infer_from_failed_alignment(self) -> None:
        with self.assertRaises(CNNAttentionScorerError):
            score_aligned_audio_context(
                "not-used.wav",
                {"alignment_status": "failed", "segments": [], "error": {"code": "textgrid_invalid"}},
            )

    def test_context_scorer_prepares_webm_once_and_passes_wav_to_mfa_and_cnn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_webm = root / "source.webm"
            prepared_wav = root / "prepared.wav"
            write_tone(source_webm)
            write_tone(prepared_wav)
            prepared = PreparedMfaAudio(prepared_wav, 1.0, 16000, 16000, 0.2, 0.1)
            alignment = {
                "status": "success",
                "alignment_status": "success",
                "method": "mfa",
                "segments": [{"index": 0, "type": "phone", "phone": "B", "start": 0.1, "end": 0.2}],
            }
            with patch("app.scorers.cnn_attention_scorer.prepare_audio_for_mfa", return_value=prepared), patch(
                "app.alignment.alignment_service.align_audio", return_value=alignment
            ) as align_mock, patch(
                "app.scorers.cnn_attention_scorer.score_aligned_audio_context", return_value={"status": "completed"}
            ) as score_mock:
                result = cnn_attention_scorer.score_pronunciation_context(
                    {"audio_path": str(source_webm), "target_word": "book", "job_id": "job-1"}
                )
        self.assertEqual(result["status"], "completed")
        self.assertIs(align_mock.call_args.kwargs["prepared_audio"], prepared)
        self.assertEqual(Path(score_mock.call_args.args[0]), prepared_wav)
        self.assertNotEqual(Path(score_mock.call_args.args[0]), source_webm)

    def test_segment_scorer_rejects_missing_or_non_wav_prepared_audio(self) -> None:
        with self.assertRaises(CNNAttentionScorerError):
            cnn_attention_scorer._load_prepared_wav("missing-original.webm")  # noqa: SLF001

    def test_context_segments_load_prepared_waveform_once(self) -> None:
        import numpy as np

        waveform = np.zeros(16000, dtype=np.float32)
        segments = [
            {"index": 0, "phone": "B", "start": 0.1, "end": 0.2},
            {"index": 1, "phone": "UH", "start": 0.2, "end": 0.3},
            {"index": 2, "phone": "K", "start": 0.3, "end": 0.4},
        ]
        prediction = {
            "predicted_error_type": "substitution",
            "class_probabilities": {"substitution": 0.9},
            "diagnosis_confidence": 0.9,
            "audio": {"context": {}},
        }
        with patch("app.scorers.cnn_attention_scorer._load_context_model", return_value=(object(), {0: "substitution"}, Path("model.pt"), "cpu")), patch(
            "app.scorers.cnn_attention_scorer._load_prepared_wav", return_value=(waveform, 16000)
        ) as load_mock, patch(
            "app.scorers.cnn_attention_scorer._predict_context_with_model", return_value=prediction
        ):
            results = cnn_attention_scorer.predict_context_segments("prepared.wav", segments)
        self.assertEqual(len(results), 3)
        self.assertEqual(load_mock.call_count, 1)

    def test_problem_phonemes_and_segments_are_ordered_by_alignment_time(self) -> None:
        predictions = [
            {"index": 2, "phone": "B", "word": "book", "start": 0.5, "end": 0.7, "predicted_error_type": "deletion", "diagnosis_confidence": 0.7, "class_probabilities": {"deletion": 0.7}},
            {"index": 1, "phone": "UH", "word": "book", "start": 0.3, "end": 0.5, "predicted_error_type": "substitution", "diagnosis_confidence": 0.9, "class_probabilities": {"substitution": 0.9}},
            {"index": 0, "phone": "K", "word": "book", "start": 0.1, "end": 0.3, "predicted_error_type": "addition", "diagnosis_confidence": 0.8, "class_probabilities": {"addition": 0.8}},
        ]
        hybrid = {
            "primary_error_type": "deletion",
            "top_issues": [
                {"phone": "B", "word": "book", "start": 0.5, "end": 0.7, "predicted_error_type": "deletion", "diagnosis_confidence": 0.7, "class_probabilities": {"deletion": 0.7}},
                {"phone": "UH", "word": "book", "start": 0.3, "end": 0.5, "predicted_error_type": "substitution", "diagnosis_confidence": 0.9, "class_probabilities": {"substitution": 0.9}},
                {"phone": "K", "word": "book", "start": 0.1, "end": 0.3, "predicted_error_type": "addition", "diagnosis_confidence": 0.8, "class_probabilities": {"addition": 0.8}},
            ],
            "feedback": {"summary": "test", "tips": []},
            "severity": "high",
            "hybrid_method": "cnn_attention_plus_heuristic_scoring",
            "hybrid_status": "success",
            "location_reliability": "forced_alignment",
        }
        scoring = {"utterance_segmental_score": 70.0, "scoring_method": "heuristic_gop", "scoring_status": "heuristic", "metadata": {"is_heuristic": True}}
        alignment = {"status": "success", "method": "mfa", "metadata": {}, "words": [], "phones": []}
        with patch("app.scoring.scoring_service.score_pronunciation_segments", return_value=scoring), patch(
            "app.hybrid.hybrid_diagnosis.build_hybrid_diagnosis", return_value=hybrid
        ):
            result = cnn_attention_scorer._aggregate_segment_predictions(predictions, alignment_result=alignment)  # noqa: SLF001
        self.assertEqual(result["predicted_error_type"], "deletion")
        self.assertEqual([segment["phone"] for segment in result["segments"]], ["K", "UH", "B"])
        self.assertEqual(result["problem_phonemes"], ["K", "UH", "B"])
        self.assertEqual(
            result["metadata"]["global_diagnosis_selection"],
            "hybrid_severity_then_heuristic_phone_score_then_classifier_diagnosis_confidence",
        )
        payload = build_success_webhook_payload("job-1", result)
        self.assertNotIn("prepared.wav", str(payload))

    def test_valid_textgrid_parses_phone_word_and_duration(self) -> None:
        result = parse_textgrid(FIXTURES / "valid.TextGrid")
        self.assertEqual(len(result["phones"]), 3)
        self.assertEqual(result["phones"][0]["word"], "cat")
        self.assertEqual(result["phones"][0]["duration"], 0.2)

    def test_missing_phone_tier_fails_quality_validation(self) -> None:
        result = parse_textgrid(FIXTURES / "missing_phone.TextGrid")
        quality = validate_alignment_quality(words=result["words"], phones=result["phones"], audio_duration=1.0, expected_word_count=1)
        self.assertEqual(quality["status"], "failed")
        self.assertIn("no_phone_segments", quality["issues"])

    def test_out_of_bounds_and_zero_duration_are_rejected(self) -> None:
        result = parse_textgrid(FIXTURES / "zero_duration.TextGrid")
        quality = validate_alignment_quality(words=result["words"], phones=result["phones"], audio_duration=0.4, expected_word_count=1)
        self.assertEqual(quality["status"], "failed")
        self.assertIn("zero_duration_phone", quality["issues"])
        self.assertIn("boundary_out_of_bounds", quality["issues"])

    def test_low_coverage_is_a_warning(self) -> None:
        quality = validate_alignment_quality(
            words=[{"start": 0.03, "end": 0.16, "word": "cat"}],
            phones=[{"start": 0.03, "end": 0.16, "phone": "K"}],
            audio_duration=0.25,
            expected_word_count=1,
        )
        self.assertEqual(quality["status"], "warning")
        self.assertEqual(quality["decision"], "valid_with_warning")
        self.assertIn("raw_audio_coverage_low", quality["warnings"])

    def test_single_word_leading_and_trailing_silence_is_usable_with_warning(self) -> None:
        words = [{"start": 0.76, "end": 1.43, "word": "family"}]
        phones = [
            {"start": 0.76, "end": 0.92, "phone": "f"},
            {"start": 0.92, "end": 1.01, "phone": "ae"},
            {"start": 1.01, "end": 1.05, "phone": "m"},
            {"start": 1.05, "end": 1.22, "phone": "l"},
            {"start": 1.22, "end": 1.43, "phone": "i"},
        ]
        quality = validate_alignment_quality(
            words=words,
            phones=phones,
            audio_duration=1.68,
            expected_word_count=1,
            expected_words=["family"],
        )
        self.assertEqual(quality["status"], "warning")
        self.assertEqual(quality["decision"], "valid_with_warning")
        self.assertEqual(quality["metrics"]["raw_audio_coverage_ratio"], 0.399)
        self.assertEqual(quality["metrics"]["phone_span_fill_ratio"], 1.0)
        # Deprecated alias remains available to existing local consumers.
        self.assertEqual(quality["metrics"]["speech_relative_coverage_ratio"], 1.0)
        self.assertIn("raw_audio_coverage_below_minimum", quality["warnings"])
        self.assertIn("leading_silence_high", quality["warnings"])
        self.assertEqual(quality["failures"], [])

    def test_few_milliseconds_of_alignment_is_invalid(self) -> None:
        quality = validate_alignment_quality(
            words=[{"start": 0.10, "end": 0.13, "word": "cat"}],
            phones=[{"start": 0.10, "end": 0.13, "phone": "K"}],
            audio_duration=1.0,
            expected_word_count=1,
            expected_words=["cat"],
        )
        self.assertEqual(quality["status"], "failed")
        self.assertIn("aligned_duration_too_short", quality["failures"])

    def test_overlapping_phones_are_invalid(self) -> None:
        quality = validate_alignment_quality(
            words=[{"start": 0.1, "end": 0.6, "word": "cat"}],
            phones=[
                {"start": 0.1, "end": 0.4, "phone": "K"},
                {"start": 0.35, "end": 0.6, "phone": "AE"},
            ],
            audio_duration=0.7,
            expected_word_count=1,
        )
        self.assertEqual(quality["status"], "failed")
        self.assertIn("overlapping_phone_segments", quality["failures"])

    def test_clean_high_coverage_alignment_is_valid(self) -> None:
        quality = validate_alignment_quality(
            words=[{"start": 0.0, "end": 0.8, "word": "cat"}],
            phones=[
                {"start": 0.0, "end": 0.3, "phone": "K"},
                {"start": 0.3, "end": 0.55, "phone": "AE"},
                {"start": 0.55, "end": 0.8, "phone": "T"},
            ],
            audio_duration=0.8,
            expected_word_count=1,
            expected_words=["cat"],
        )
        self.assertEqual(quality["status"], "ok")
        self.assertEqual(quality["decision"], "valid")


@unittest.skipUnless(os.getenv("RUN_MFA_INTEGRATION_TESTS") == "1", "set RUN_MFA_INTEGRATION_TESTS=1 to run MFA locally")
class MfaIntegrationTest(unittest.TestCase):
    def test_local_mfa_alignment(self) -> None:
        audio = os.getenv("MFA_INTEGRATION_AUDIO")
        text = os.getenv("MFA_INTEGRATION_TEXT")
        self.assertTrue(audio, "MFA_INTEGRATION_AUDIO is required")
        self.assertTrue(text, "MFA_INTEGRATION_TEXT is required")
        result = run_mfa_alignment(audio, text)
        self.assertIn(result["alignment_status"], {"success", "warning"})
        self.assertGreater(len(result["phones"]), 0)


if __name__ == "__main__":
    unittest.main()
