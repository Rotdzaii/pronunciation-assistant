from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app.alignment.audio_preparation import PreparedMfaAudio, prepare_audio_for_mfa
from app.alignment.alignment_service import align_audio
from app.alignment.mfa_aligner import _mfa_command_prefix, run_mfa_alignment
from app.alignment.quality import validate_alignment_quality
from app.alignment.textgrid_parser import parse_textgrid
from app.alignment.transcript import normalize_and_check_transcript
from app.contracts.alignment_contract import AlignmentError
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


class MfaAlignmentUnitTests(unittest.TestCase):
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

    def test_context_scorer_does_not_infer_from_failed_alignment(self) -> None:
        with self.assertRaises(CNNAttentionScorerError):
            score_aligned_audio_context(
                "not-used.wav",
                {"alignment_status": "failed", "segments": [], "error": {"code": "textgrid_invalid"}},
            )

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
            words=[{"start": 0.1, "end": 0.2, "word": "cat"}],
            phones=[{"start": 0.1, "end": 0.2, "phone": "K"}],
            audio_duration=0.2,
            expected_word_count=1,
        )
        self.assertEqual(quality["status"], "warning")
        self.assertIn("speech_coverage_low", quality["issues"])


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
