"""
Unit tests for app.phonetics.canonicalization.

Covers:
  1. glass  — allophones ɟ→ɡ, ʎ→l
  2. rice   — diphthong aj→aɪ, ɹ→r
  3. book   — no mapping needed (canonical identity)
  4. substitution with phone NOT in expected_phones → diagnosis_invalid
  5. addition outside inventory → not flagged as diagnosis_invalid
  6. forced_alignment=False → reliability invalid (no_forced_alignment)
  7. valid_with_warning (silence/quality warning) → CNN score kept
  8. webhook-like payload contains no local path / TextGrid / signed URL / secret
"""

import unittest
from app.phonetics.canonicalization import (
    canonicalize_phones,
    check_inventory_guard,
    compute_reliability,
    canonicalize_phone,
    normalize_alignment_phones,
    normalize_expected_phones,
    CANONICAL_MFA_PHONES,
    inventory_audit,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _alignment_result(
    phones: list[str],
    *,
    is_forced: bool = True,
    method: str = "mfa",
    quality_warnings: list[str] | None = None,
    quality_failures: list[str] | None = None,
) -> dict:
    return {
        "method": method,
        "status": "success" if is_forced else "failed",
        "phones": [{"phone": p, "start": i * 0.1, "end": (i + 1) * 0.1} for i, p in enumerate(phones)],
        "words": [],
        "metadata": {
            "is_forced_alignment": is_forced,
            "mfa_used": is_forced,
            "alignment_quality_warnings": quality_warnings or [],
            "alignment_quality_failures": quality_failures or [],
            "fallback_alignment": not is_forced,
            "is_fallback": not is_forced,
        },
    }


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCanonicalizePhone(unittest.TestCase):

    # 1. glass — ɟ→ɡ, ʎ→l
    def test_glass_allophones(self):
        raw = ["ɟ", "ʎ", "æ", "s"]
        result = canonicalize_phones(raw)
        self.assertEqual(result["canonical_phones"], ["ɡ", "l", "æ", "s"])
        self.assertEqual(result["display_pronunciation"], "ɡlæs")
        self.assertFalse(result["has_unknown"])
        self.assertFalse(result["has_ambiguous"])

    # 2. rice — canonical IPA ɹ, aj→aɪ
    def test_rice_allophone_and_diphthong(self):
        raw = ["ɹ", "aj", "s"]
        result = canonicalize_phones(raw)
        self.assertEqual(result["canonical_phones"], ["ɹ", "aɪ", "s"])
        self.assertEqual(result["display_pronunciation"], "ɹaɪs")

    # 3. book — identity mapping
    def test_book_identity(self):
        raw = ["b", "ʊ", "k"]
        result = canonicalize_phones(raw)
        self.assertEqual(result["canonical_phones"], ["b", "ʊ", "k"])
        self.assertEqual(result["display_pronunciation"], "bʊk")
        # All three phones are identity mappings
        for detail in result["details"]:
            self.assertEqual(detail["mapping"], "identity")

    # 4. substitution with problem phone NOT in expected_phones → diagnosis_invalid
    def test_inventory_guard_rejects_substitution_outside_inventory(self):
        expected = ["b", "ʊ", "k"]
        problem_phone = "ɑ"  # not in expected
        diagnosis_invalid, reason = check_inventory_guard(problem_phone, "substitution", expected)
        self.assertTrue(diagnosis_invalid)
        self.assertIn("ɑ", reason)

    # 5. addition outside inventory → NOT flagged as diagnosis_invalid
    def test_inventory_guard_allows_addition_outside_inventory(self):
        expected = ["b", "ʊ", "k"]
        problem_phone = "ɑ"  # not in expected, but addition is exempt
        diagnosis_invalid, reason = check_inventory_guard(problem_phone, "addition", expected)
        self.assertFalse(diagnosis_invalid)
        self.assertIsNone(reason)

    def test_inventory_guard_rejects_unsupported_addition(self):
        diagnosis_invalid, reason = check_inventory_guard("not-a-phone", "addition", ["b", "ʊ", "k"])
        self.assertTrue(diagnosis_invalid)
        self.assertEqual(reason, "problem_phone_not_canonicalizable")

    # 6. forced_alignment=False → reliability invalid
    def test_no_forced_alignment_yields_invalid_reliability(self):
        alignment = _alignment_result(["b", "ʊ", "k"], is_forced=False, method="fallback_even_split")
        canon = canonicalize_phones(["b", "ʊ", "k"])
        reliability = compute_reliability(alignment, canon)
        self.assertEqual(reliability["status"], "invalid")
        self.assertIn("no_forced_alignment", reliability["failures"])

    # 7. valid_with_warning — forced alignment OK, but silence quality warning
    def test_silence_warning_yields_valid_with_warning(self):
        alignment = _alignment_result(
            ["b", "ʊ", "k"],
            is_forced=True,
            method="mfa",
            quality_warnings=["leading_silence_high"],
        )
        canon = canonicalize_phones(["b", "ʊ", "k"])
        reliability = compute_reliability(alignment, canon)
        self.assertEqual(reliability["status"], "valid_with_warning")
        self.assertIn("leading_silence_high", reliability["warnings"])
        self.assertTrue(reliability["forced_alignment"])
        # Score should NOT be suppressed (only invalid suppresses score)
        self.assertEqual(reliability["failures"], [])

    # 8. no sensitive local paths or tokens in canonicalization output
    def test_no_sensitive_data_in_canonicalization_output(self):
        raw = ["ɟ", "ʎ", "æ", "s"]
        result = canonicalize_phones(raw)
        payload_str = str(result).lower()
        sensitive_markers = [
            "/tmp/", "c:\\", "appdata\\local\\temp",
            ".textgrid", ".wav",
            "x-amz-signature=", "token=", "sig=",
        ]
        for marker in sensitive_markers:
            self.assertNotIn(marker, payload_str, f"Sensitive marker found: {marker!r}")


class TestCanonicalizePhoneEdgeCases(unittest.TestCase):

    def test_silence_phones_excluded_from_canonical_list(self):
        raw = ["sp", "b", "ʊ", "k", "sp"]
        result = canonicalize_phones(raw)
        self.assertEqual(result["canonical_phones"], ["b", "ʊ", "k"])
        silence_details = [d for d in result["details"] if d["mapping"] == "silence"]
        self.assertEqual(len(silence_details), 2)

    def test_ambiguous_flap_flagged_without_context(self):
        result = canonicalize_phone("ɾ")
        self.assertEqual(result["mapping"], "ambiguous")
        self.assertIsNone(result["canonical"])
        self.assertIn("ambiguous_flap", result["canonicalization_warning"])

    def test_ambiguous_flap_is_not_guessed_from_expected_phone(self):
        result = canonicalize_phone("ɾ", expected_at_position="t")
        self.assertEqual(result["mapping"], "ambiguous")
        self.assertIsNone(result["canonical"])

    def test_unsupported_phone_is_not_emitted_as_canonical_output(self):
        result = canonicalize_phones(["ɟ", "ʎ", "aj", "not-a-phone"])
        self.assertEqual(result["canonical_phones"], ["ɡ", "l", "aɪ"])
        self.assertNotIn("ɟ", result["display_pronunciation"])
        self.assertNotIn("ʎ", result["display_pronunciation"])
        self.assertNotIn("aj", result["display_pronunciation"])
        self.assertTrue(result["has_unsupported"])

    def test_english_mfa_inventory_audit_is_complete_and_classified(self):
        audit = inventory_audit()
        self.assertEqual(audit["total"], 100)
        self.assertIn("ɹ", audit["pass_through"])
        self.assertIn("ɟ", audit["safe_mapping"])
        self.assertIn("ɾ", audit["ambiguous"])
        self.assertIn("ɚ", audit["pass_through"])
        self.assertIn("tʲ", audit["safe_mapping"])
        self.assertEqual(len(audit["pass_through"]), 36)
        self.assertEqual(len(audit["safe_mapping"]), 13)
        self.assertEqual(len(audit["ambiguous"]), 1)
        self.assertEqual(len(audit["unsupported"]), 50)
        self.assertIn("ɲ", audit["unsupported"])

    def test_all_canonical_phones_are_identity(self):
        for phone in CANONICAL_MFA_PHONES:
            result = canonicalize_phone(phone)
            self.assertEqual(result["mapping"], "identity", f"Expected identity for {phone!r}")
            self.assertEqual(result["canonical"], phone)

    def test_aspirated_stops(self):
        for raw, expected in [("kʰ", "k"), ("tʰ", "t"), ("pʰ", "p")]:
            result = canonicalize_phone(raw)
            self.assertEqual(result["canonical"], expected)
            self.assertEqual(result["mapping"], "allophone")

    def test_diphthongs_normalized(self):
        for raw, expected in [("aj", "aɪ"), ("aw", "aʊ"), ("ej", "eɪ"), ("ow", "oʊ"), ("ɔj", "ɔɪ")]:
            result = canonicalize_phone(raw)
            self.assertEqual(result["canonical"], expected)


class TestTeacherRhoticCanonicalization(unittest.TestCase):

    def test_mfa_tj_composite_maps_to_t(self):
        result = canonicalize_phone("tj")
        self.assertEqual(result["canonical"], "t")
        self.assertEqual(result["mapping"], "allophone")

    def test_rhotic_vowel_is_a_supported_canonical_phone(self):
        result = canonicalize_phone("ɚ")
        self.assertEqual(result["canonical"], "ɚ")
        self.assertEqual(result["mapping"], "identity")

    def test_en_us_expected_rhotic_sequence_collapses_to_one_phone(self):
        self.assertEqual(
            normalize_expected_phones(["t", "iː", "tʃ", "ə", "r"], variant="en-US"),
            ["t", "iː", "tʃ", "ɚ"],
        )
        self.assertEqual(
            normalize_expected_phones(["t", "iː", "tʃ", "ə", "ɹ"], variant="en-US"),
            ["t", "iː", "tʃ", "ɚ"],
        )

    def test_non_en_us_expected_rhotic_sequence_is_not_collapsed(self):
        self.assertEqual(
            normalize_expected_phones(["ə", "r"], variant="en-GB"),
            ["ə", "ɹ"],
        )

    def test_teacher_alignment_keeps_rhotic_timing_segment_and_is_reliable_with_warnings(self):
        raw_alignment = ["tj", "iː", "tʃ", "ɚ"]
        expected = normalize_expected_phones(["t", "iː", "tʃ", "ə", "r"], variant="en-US")
        canonical = canonicalize_phones(raw_alignment)
        self.assertEqual(normalize_alignment_phones(raw_alignment), expected)
        self.assertEqual(len(canonical["details"]), 4)
        self.assertEqual(canonical["details"][3]["canonical"], "ɚ")

        alignment = _alignment_result(
            raw_alignment,
            quality_warnings=[
                "raw_audio_coverage_below_minimum",
                "leading_silence_high",
                "trailing_silence_high",
            ],
        )
        invalid, reason = check_inventory_guard("tj", "substitution", expected)
        self.assertFalse(invalid)
        self.assertIsNone(reason)
        reliability = compute_reliability(alignment, canonical)
        self.assertEqual(reliability["status"], "valid_with_warning")
        self.assertEqual(reliability["failures"], [])
        self.assertNotIn("unsupported_phones_in_alignment", reliability["failures"])
        self.assertNotIn("problem_phone_not_canonicalizable", reliability["failures"])


if __name__ == "__main__":
    unittest.main()
