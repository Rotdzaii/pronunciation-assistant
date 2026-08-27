from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_l2_arctic_expected_observed_v4 import (  # noqa: E402
    canonicalize_phone,
    derive_label,
)


class ExpectedObservedContractTests(unittest.TestCase):
    def test_plain_consonant_is_clean_correct(self) -> None:
        result = derive_label("TH")
        self.assertEqual(result.raw_label, "TH")
        self.assertEqual(result.relation, "correct")
        self.assertEqual(result.expected_phone_raw, "TH")
        self.assertEqual(result.observed_phone_raw, "TH")
        self.assertEqual(result.expected_phone_canonical, "TH")
        self.assertEqual(result.observed_phone_canonical, "TH")
        self.assertEqual(result.label_quality, "clean")

    def test_stressed_vowel_is_canonicalized_only_in_derived_fields(self) -> None:
        result = derive_label("AH1")
        self.assertEqual(result.raw_label, "AH1")
        self.assertEqual(result.expected_phone_raw, "AH1")
        self.assertEqual(result.observed_phone_raw, "AH1")
        self.assertEqual(result.expected_phone_canonical, "AH")
        self.assertEqual(result.observed_phone_canonical, "AH")
        self.assertEqual(canonicalize_phone("AH1"), "AH")

    def test_valid_substitution(self) -> None:
        result = derive_label("TH,T,s")
        self.assertEqual(result.relation, "substitution")
        self.assertEqual(result.expected_phone_raw, "TH")
        self.assertEqual(result.observed_phone_raw, "T")
        self.assertEqual(result.research_subset, "PHONE_IDENTIFICATION_ELIGIBLE")

    def test_substitution_stress_canonicalization(self) -> None:
        result = derive_label("IH1,IY,s")
        self.assertEqual(result.expected_phone_raw, "IH1")
        self.assertEqual(result.observed_phone_raw, "IY")
        self.assertEqual(result.expected_phone_canonical, "IH")
        self.assertEqual(result.observed_phone_canonical, "IY")

    def test_valid_deletion_is_isolated(self) -> None:
        result = derive_label("T,sil,d")
        self.assertEqual(result.relation, "deletion")
        self.assertEqual(result.expected_phone_raw, "T")
        self.assertEqual(result.observed_phone_raw, "<SIL>")
        self.assertEqual(result.observed_phone_canonical, "<SIL>")
        self.assertTrue(result.is_deletion)
        self.assertTrue(result.is_deletion_eligible)
        self.assertEqual(result.research_subset, "DELETION_ELIGIBLE")

    def test_valid_addition_is_isolated(self) -> None:
        result = derive_label("sil,T,a")
        self.assertEqual(result.relation, "addition")
        self.assertEqual(result.expected_phone_raw, "sil")
        self.assertEqual(result.expected_phone_canonical, "<SIL>")
        self.assertEqual(result.observed_phone_raw, "T")
        self.assertEqual(result.label_quality, "excluded_addition")
        self.assertEqual(result.research_subset, "ADDITION_AUDIT")
        self.assertFalse(result.is_main_clean)

    def test_malformed_plain_labels_are_unresolved(self) -> None:
        for label in ("R*", "err", "w", "Ah", "ERR"):
            with self.subTest(label=label):
                result = derive_label(label)
                self.assertEqual(result.raw_label, label)
                self.assertEqual(result.relation, "unresolved")
                self.assertEqual(result.label_quality, "unresolved")

    def test_invalid_observed_substitution_is_unresolved(self) -> None:
        result = derive_label("TH,R*,s")
        self.assertEqual(result.relation, "unresolved")
        self.assertEqual(result.tagged_relation, "substitution")
        self.assertEqual(result.expected_phone_raw, "TH")
        self.assertEqual(result.observed_phone_raw, "R*")
        self.assertIn("invalid_observed", result.exclusion_reason)

    def test_invalid_lowercase_observed_phone_is_unresolved(self) -> None:
        result = derive_label("TH,w,s")
        self.assertEqual(result.relation, "unresolved")
        self.assertEqual(result.observed_phone_raw, "w")

    def test_non_speech_is_explicitly_excluded(self) -> None:
        for label in ("", "sp", "sil"):
            with self.subTest(label=label):
                result = derive_label(label)
                self.assertEqual(result.relation, "non_speech")
                self.assertEqual(result.label_quality, "excluded_non_speech")

    def test_raw_label_whitespace_is_never_mutated(self) -> None:
        raw = "  AH1  "
        result = derive_label(raw)
        self.assertEqual(result.raw_label, raw)
        self.assertEqual(result.expected_phone_raw, "AH1")
        self.assertEqual(result.expected_phone_canonical, "AH")

    def test_malformed_triplets_do_not_crash(self) -> None:
        for label in (None, "AH0,,", "TH,T,s,extra", ",,s", ","):
            with self.subTest(label=label):
                result = derive_label(label)
                self.assertIn(result.relation, {"non_speech", "unresolved"})


if __name__ == "__main__":
    unittest.main()
