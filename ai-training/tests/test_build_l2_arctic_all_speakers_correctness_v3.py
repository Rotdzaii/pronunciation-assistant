from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_l2_arctic_all_speakers_correctness_v3 import (  # noqa: E402
    classify_label,
    is_plain_arpabet_phone,
    parse_phone_intervals,
)


class LabelMappingTests(unittest.TestCase):
    def assert_mapping(self, label: object, expected: str) -> None:
        self.assertEqual(classify_label(label).error_type, expected)

    def test_plain_phones_are_correct(self) -> None:
        for label in ("N", "AH0", "IH1", "TH"):
            with self.subTest(label=label):
                self.assert_mapping(label, "correct")

    def test_error_annotations(self) -> None:
        cases = {
            "TH,T,s": "substitution",
            "T,sil,d": "deletion",
            "sil,AH0,a": "addition",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                result = classify_label(label)
                self.assertEqual(result.error_type, expected)
                self.assertNotEqual(result.error_type, "correct")
        self.assertEqual(classify_label("TH,T,s").expected_phone, "TH")
        self.assertEqual(classify_label("T,sil,d").expected_phone, "T")
        self.assertEqual(classify_label("sil,AH0,a").expected_phone, "sil")

    def test_non_speech(self) -> None:
        for label in ("", "sp", "sil"):
            with self.subTest(label=label):
                self.assert_mapping(label, "non_speech")

    def test_known_residuals_remain_unknown(self) -> None:
        for label in ("spn", "D_", "s"):
            with self.subTest(label=label):
                self.assert_mapping(label, "unknown")

    def test_vowel_stress_zero_one_two_is_valid(self) -> None:
        for label in ("AH0", "AH1", "AH2", "IH0", "IH1", "IH2"):
            with self.subTest(label=label):
                self.assertTrue(is_plain_arpabet_phone(label))
                self.assert_mapping(label, "correct")

    def test_plain_phone_never_becomes_error(self) -> None:
        for label in ("N", "T", "R", "TH", "AE1"):
            with self.subTest(label=label):
                self.assertNotIn(
                    classify_label(label).error_type,
                    {"substitution", "deletion", "addition"},
                )

    def test_malformed_labels_do_not_crash(self) -> None:
        cases = {
            None: "non_speech",
            "AH0,,": "unknown",
            "TH,T,s,extra": "unknown",
            ",,s": "unknown",
            ",": "unknown",
            "V``": "unknown",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assert_mapping(label, expected)


class TextGridParserTests(unittest.TestCase):
    def test_only_phone_interval_tier_is_returned(self) -> None:
        lines = [
            'class = "IntervalTier"',
            'name = "words"',
            "intervals [1]:",
            "xmin = 0",
            "xmax = 1",
            'text = "word"',
            'class = "IntervalTier"',
            'name = "phones"',
            "intervals [1]:",
            "xmin = 0",
            "xmax = 0.5",
            'text = "AH0"',
            "intervals [2]:",
            "xmin = 0.5",
            "xmax = 1",
            'text = "D_"',
        ]
        intervals = parse_phone_intervals(lines, "TEST", "utterance")
        self.assertEqual([interval.label for interval in intervals], ["AH0", "D_"])
        self.assertEqual(classify_label(intervals[1].label).error_type, "unknown")

    def test_malformed_numeric_interval_is_skipped_without_crash(self) -> None:
        lines = [
            'class = "IntervalTier"',
            'name = "phones"',
            "intervals [1]:",
            "xmin = not-a-number",
            "xmax = 1",
            'text = "AH0"',
        ]
        self.assertEqual(parse_phone_intervals(lines, "TEST", "utterance"), [])


if __name__ == "__main__":
    unittest.main()
