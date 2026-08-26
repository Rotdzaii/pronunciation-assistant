"""
ai-worker/app/phonetics/canonicalization.py

Single source of truth for phone canonicalization and result-level reliability.

Responsibilities:
  1. Map raw MFA allophone/allographic phones → canonical IPA (unambiguous).
  2. Map ARPABET (with or without stress digits) → canonical IPA.
  3. Flag ambiguous phones that cannot be safely canonicalized without context.
  4. Compute reliability status (valid / valid_with_warning / invalid) from
     alignment metadata and diagnosis guard results.
  5. Build phone_results list with canonical phones for frontend consumption.
  6. Enforce the diagnosis inventory guard: substitution/deletion phones must
     appear in the expected canonical phone set.

Phone inventory basis: english_mfa acoustic model + english_mfa dictionary.
Pronunciation variant: en-US.
"""

from __future__ import annotations

from typing import Any


# ── MFA allophone → canonical IPA (unambiguous only) ─────────────────────────

_MFA_ALLOPHONE_TO_CANONICAL: dict[str, str] = {
    # Palatalized voiced velar plosive (before front vowels in english_mfa)
    "ɟ":  "ɡ",
    # Palatalized lateral approximant
    "ʎ":  "l",
    # Velarized (dark) lateral — syllable-final / pre-consonant position
    "ɫ":  "l",
    "ɫ̩": "l",   # syllabic velarized lateral (e.g. "bottle")
    # Aspirated stops — canonical IPA for English drops the aspiration diacritic
    "pʰ": "p",
    "tʰ": "t",
    "kʰ": "k",
    # english_mfa emits a palatalized /t/ as tʲ; some TextGrid exports spell
    # the same composite label as ASCII "tj". Both are timing labels for /t/,
    # not a rule for any one word.
    "tʲ": "t",
    "tj": "t",
    # Pre-normalized diphthong representations found in some MFA dictionaries
    "i": "iː",
    "aj": "aɪ",
    "aw": "aʊ",
    "ej": "eɪ",
    "ow": "oʊ",
    "ɔj": "ɔɪ",
}

# Phones whose context-free canonicalization is AMBIGUOUS.
# These are flagged with canonicalization_warning; do not silently guess.
_AMBIGUOUS_PHONES: dict[str, str] = {
    # American English alveolar flap between vowels: may be /t/ (butter) or /d/ (ladder)
    "ɾ": "ambiguous_flap_td",
}

# ── Canonical phone set for english_mfa ──────────────────────────────────────
# Phones already in this set pass through unchanged.
CANONICAL_MFA_PHONES: frozenset[str] = frozenset({
    # Monophthong vowels
    "iː", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɑː", "ɔ", "ɔː", "oʊ",
    "ʊ", "uː", "ʌ", "ɜː", "ə", "ɚ",
    # Diphthongs (canonical forms)
    "aɪ", "aʊ", "ɔɪ",
    # Plosives
    "p", "b", "t", "d", "k", "ɡ",
    # Affricates
    "tʃ", "dʒ",
    # Fricatives
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    # Nasals
    "m", "n", "ŋ",
    # Approximants / liquids. This contract uses canonical IPA, so the
    # American English rhotic is represented as /ɹ/, not the display-only "r".
    "l", "ɹ", "j", "w",
})

SILENCE_PHONES: frozenset[str] = frozenset({"sp", "spn", "sil", "SIL", "SP", "SPN", ""})

# ── ARPABET → canonical IPA ────────────────────────────────────────────────────
# AH0→ə and ER0→ər are semantically distinct from their stressed forms.
_ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ",   "AE": "æ",   "AH": "ʌ",   "AH0": "ə",
    "AO": "ɔ",   "AW": "aʊ",  "AY": "aɪ",
    "EH": "ɛ",   "ER": "ɜɹ", "ER0": "əɹ", "EY": "eɪ",
    "IH": "ɪ",   "IY": "iː",  "OW": "oʊ",  "OY": "ɔɪ",
    "UH": "ʊ",   "UW": "uː",
    "B":  "b",   "CH": "tʃ",  "D":  "d",   "DH": "ð",
    "F":  "f",   "G":  "ɡ",   "HH": "h",   "JH": "dʒ",
    "K":  "k",   "L":  "l",   "M":  "m",   "N":  "n",
    "NG": "ŋ",   "P":  "p",   "R":  "ɹ",   "S":  "s",
    "SH": "ʃ",   "T":  "t",   "TH": "θ",   "V":  "v",
    "W":  "w",   "WH": "ʍ",   "Y":  "j",   "Z":  "z",
    "ZH": "ʒ",
}

# Exact non-silence inventory from english_mfa acoustic model metadata
# (model version 3.1.0, distributed with MFA 3.4.0).  Silence markers remain
# separate because they are not pronunciation phones.  A phone may be known to
# the model yet still be unsupported by this public canonicalization contract.
ENGLISH_MFA_PHONE_INVENTORY: frozenset[str] = frozenset({
    "a", "aj", "aw", "aː", "b", "bʲ", "c", "cʰ", "cʷ", "d", "dʒ", "dʲ", "d̪",
    "e", "ej", "eː", "f", "fʲ", "fʷ", "h", "i", "iː", "j", "k", "kp", "kʰ", "kʷ",
    "l", "m", "mʲ", "m̩", "n", "n̩", "o", "ow", "oː", "p", "pʰ", "pʲ", "pʷ", "s",
    "t", "tʃ", "tʰ", "tʲ", "tʷ", "t̪", "u", "uː", "v", "vʲ", "vʷ", "w", "z", "æ",
    "ç", "ð", "ŋ", "ɐ", "ɑ", "ɑː", "ɒ", "ɒː", "ɔ", "ɔj", "ɖ", "ə", "əw", "ɚ", "ɛ",
    "ɛː", "ɜ", "ɜː", "ɝ", "ɟ", "ɟʷ", "ɡ", "ɡb", "ɡʷ", "ɪ", "ɫ", "ɫ̩", "ɱ", "ɲ", "ɹ",
    "ɾ", "ɾʲ", "ɾ̃", "ʃ", "ʈ", "ʈʲ", "ʈʷ", "ʉ", "ʉː", "ʊ", "ʋ", "ʎ", "ʒ", "ʔ", "θ",
})


def inventory_audit() -> dict[str, Any]:
    """Return a deterministic audit of the english_mfa phone contract.

    ``unsupported`` contains model phones deliberately not canonicalized. They
    are rejected at runtime rather than guessed or exposed as public phones.
    """
    return {
        "total": len(ENGLISH_MFA_PHONE_INVENTORY),
        "pass_through": sorted(ENGLISH_MFA_PHONE_INVENTORY & CANONICAL_MFA_PHONES),
        "safe_mapping": sorted(ENGLISH_MFA_PHONE_INVENTORY & _MFA_ALLOPHONE_TO_CANONICAL.keys()),
        "ambiguous": sorted(ENGLISH_MFA_PHONE_INVENTORY & _AMBIGUOUS_PHONES.keys()),
        "unsupported": sorted(
            ENGLISH_MFA_PHONE_INVENTORY
            - CANONICAL_MFA_PHONES
            - _MFA_ALLOPHONE_TO_CANONICAL.keys()
            - _AMBIGUOUS_PHONES.keys()
        ),
    }


def _strip_stress(code: str) -> str:
    upper = code.upper()
    if upper in ("AH0", "ER0"):
        return upper
    return upper.rstrip("012")


def arpabet_to_ipa(code: str) -> str | None:
    """Convert ARPABET phone (with or without stress digit) to canonical IPA.
    Returns None if the code is not recognized.
    """
    return _ARPABET_TO_IPA.get(_strip_stress(code)) or _ARPABET_TO_IPA.get(code.upper())


# ── Per-phone canonicalization ────────────────────────────────────────────────

def canonicalize_phone(
    phone: str,
    *,
    expected_at_position: str | None = None,
) -> dict[str, Any]:
    """Canonicalize a single phone from MFA output.

    Returns:
      {
        "canonical": str | None,          # canonical IPA; None when unsafe
        "raw": str,
        "mapping": str,                   # identity | allophone | ambiguous |
                                          #   arpabet | silence | unsupported
        "canonicalization_warning": str | None,
      }
    """
    raw = phone.strip()

    if raw in SILENCE_PHONES:
        return {"canonical": None, "raw": raw, "mapping": "silence", "canonicalization_warning": None}

    if raw in CANONICAL_MFA_PHONES:
        return {"canonical": raw, "raw": raw, "mapping": "identity", "canonicalization_warning": None}

    if raw in _MFA_ALLOPHONE_TO_CANONICAL:
        return {
            "canonical": _MFA_ALLOPHONE_TO_CANONICAL[raw],
            "raw": raw,
            "mapping": "allophone",
            "canonicalization_warning": None,
        }

    if raw in _AMBIGUOUS_PHONES:
        return {
            "canonical": None,
            "raw": raw,
            "mapping": "ambiguous",
            "canonicalization_warning": _AMBIGUOUS_PHONES[raw],
        }

    ipa = arpabet_to_ipa(raw)
    if ipa:
        return {"canonical": ipa, "raw": raw, "mapping": "arpabet", "canonicalization_warning": None}

    return {
        "canonical": None,
        "raw": raw,
        "mapping": "unsupported",
        "canonicalization_warning": f"unsupported_phone_{raw}",
    }


def canonicalize_phones(
    phones: list[str],
    *,
    expected_phones: list[str] | None = None,
) -> dict[str, Any]:
    """Canonicalize a list of raw phones from alignment output.

    Args:
        phones: Raw phone strings from alignment_result["phones"][*]["phone"].
        expected_phones: Optional positional hint list (canonical IPA) used to
                         disambiguate ambiguous allophones like ɾ.

    Returns:
      {
        "canonical_phones": list[str],      # canonical IPA, silence excluded
        "display_pronunciation": str,       # "".join(canonical_phones)
        "has_ambiguous": bool,
        "has_unknown": bool,               # Deprecated alias for has_unsupported
        "has_unsupported": bool,
        "canonicalization_warnings": list[str],
        "details": list[dict],              # per-phone result from canonicalize_phone()
      }
    """
    details: list[dict[str, Any]] = []
    canonical_list: list[str] = []
    warnings: list[str] = []

    for i, phone in enumerate(phones):
        hint = expected_phones[i] if expected_phones and i < len(expected_phones) else None
        result = canonicalize_phone(phone, expected_at_position=hint)
        details.append(result)
        if result["mapping"] == "silence":
            continue
        if result["canonical"] is not None:
            canonical_list.append(result["canonical"])
        if result["canonicalization_warning"]:
            warnings.append(result["canonicalization_warning"])

    return {
        "canonical_phones": canonical_list,
        "display_pronunciation": "".join(canonical_list),
        "has_ambiguous": any(d["mapping"] == "ambiguous" for d in details),
        "has_unknown": any(d["mapping"] == "unsupported" for d in details),
        "has_unsupported": any(d["mapping"] == "unsupported" for d in details),
        "canonicalization_warnings": warnings,
        "details": details,
    }


def normalize_alignment_phones(phones: list[str]) -> list[str]:
    """Return canonical MFA timing labels without splitting any segment."""
    return canonicalize_phones(phones)["canonical_phones"]


def normalize_expected_phones(phones: list[str], *, variant: str | None = None) -> list[str]:
    """Normalize an expected pronunciation sequence for comparison with MFA.

    American English ``ə + r/ɹ`` denotes one rhotic-vowel timing target in the
    english_mfa inventory. Collapse only that en-US sequence; elsewhere retain
    the canonical /ɹ/ representation and do not manufacture segments.
    """
    canonical: list[str] = []
    for phone in phones:
        result = canonicalize_phone(str(phone))
        normalized = result.get("canonical")
        if normalized is not None:
            canonical.append(normalized)

    normalized_variant = str(variant or "").replace("_", "-").casefold()
    if normalized_variant != "en-us":
        return canonical

    collapsed: list[str] = []
    index = 0
    while index < len(canonical):
        if canonical[index] == "ə" and index + 1 < len(canonical) and canonical[index + 1] == "ɹ":
            collapsed.append("ɚ")
            index += 2
            continue
        collapsed.append(canonical[index])
        index += 1
    return collapsed


# ── Diagnosis inventory guard ─────────────────────────────────────────────────

def check_inventory_guard(
    problem_phone: str | None,
    error_type: str | None,
    expected_phones: list[str],
) -> tuple[bool, str | None]:
    """Validate that a diagnosed problem phone is in the expected phone inventory.

    For substitution and deletion, the problem phone must appear in
    ``expected_phones``. An addition may be outside that expected sequence, but
    it must still canonicalize into the supported public inventory.

    Returns (diagnosis_invalid: bool, reason: str | None).
    """
    if not problem_phone:
        return False, None

    canonical_result = canonicalize_phone(problem_phone)
    canonical_phone = canonical_result.get("canonical")
    if canonical_phone is None:
        return True, "problem_phone_not_canonicalizable"

    if error_type == "addition":
        return False, None
    if error_type not in {"substitution", "deletion"}:
        return False, None

    expected_set = set(expected_phones)
    if canonical_phone not in expected_set:
        return True, f"problem_phone_not_in_expected_inventory:{canonical_phone}"

    return False, None


# ── Reliability computation ───────────────────────────────────────────────────

def compute_reliability(
    alignment_result: dict[str, Any],
    canonicalization_result: dict[str, Any],
    *,
    diagnosis_invalid: bool = False,
    guard_reason: str | None = None,
) -> dict[str, Any]:
    """Compute reliability status from alignment metadata and canonicalization.

    Returns:
      {
        "status": "valid" | "valid_with_warning" | "invalid",
        "forced_alignment": bool,
        "alignment_method": str | None,
        "warnings": list[str],
        "failures": list[str],
      }
    """
    metadata = alignment_result.get("metadata") or {}
    method = str(alignment_result.get("method") or "")
    is_forced = bool(
        metadata.get("is_forced_alignment")
        or (metadata.get("mfa_used") and method == "mfa")
    )
    is_fallback = (
        method.startswith("fallback")
        or bool(metadata.get("fallback_alignment"))
        or bool(metadata.get("is_fallback"))
    )

    failures: list[str] = []
    warnings: list[str] = []

    # Hard failures → invalid
    if not is_forced or is_fallback:
        failures.append("no_forced_alignment")

    if str(alignment_result.get("status") or "").lower() not in {"success", "warning"}:
        failures.append("alignment_not_successful")

    for key, failure in (
        ("audio_rejected", "audio_rejected"),
        ("inference_failed", "inference_failed"),
        ("alignment_failed", "alignment_failed"),
    ):
        if metadata.get(key):
            failures.append(failure)

    quality_failures = metadata.get("alignment_quality_failures") or []
    for f in quality_failures:
        if str(f) not in failures:
            failures.append(str(f))

    if canonicalization_result.get("has_unsupported") or canonicalization_result.get("has_unknown"):
        failures.append("unsupported_phones_in_alignment")

    if canonicalization_result.get("has_ambiguous"):
        failures.append("ambiguous_phones_in_alignment")

    if diagnosis_invalid:
        reason = guard_reason or "problem_phone_not_in_expected_inventory"
        failures.append(reason)

    if failures:
        return {
            "status": "invalid",
            "forced_alignment": is_forced and not is_fallback,
            "alignment_method": method or None,
            "warnings": warnings,
            "failures": failures,
        }

    # Soft warnings → valid_with_warning
    quality_warnings = metadata.get("alignment_quality_warnings") or []
    for w in quality_warnings:
        warnings.append(str(w))

    if warnings:
        return {
            "status": "valid_with_warning",
            "forced_alignment": True,
            "alignment_method": method or None,
            "warnings": warnings,
            "failures": [],
        }

    return {
        "status": "valid",
        "forced_alignment": True,
        "alignment_method": method or None,
        "warnings": [],
        "failures": [],
    }


# ── Phone results list ────────────────────────────────────────────────────────

def build_phone_results(
    alignment_result: dict[str, Any],
    segment_predictions: list[dict[str, Any]],
    canonicalization_result: dict[str, Any],
    *,
    reliability_status: str | None = None,
) -> list[dict[str, Any]]:
    """Build canonical phone_results list suitable for frontend display.

    Matches each aligned phone (in temporal order, canonical IPA) against the
    segment_predictions to determine per-phone status and error type.
    Silence phones are excluded.

    Returns list of:
      {
        "phone": str,                       # canonical IPA
        "status": "ok" | "needs_improvement" | "unknown",
        "error_type": str | None,
        "start_time": float | None,
        "end_time": float | None,
        "diagnosis_confidence": float | None,
      }
    """
    alignment_phones = alignment_result.get("phones") or []
    details = canonicalization_result.get("details") or []

    # Build a lookup: (phone, round(start,2)) → prediction for fast matching
    pred_lookup: dict[tuple[str | None, float | None], dict[str, Any]] = {}
    for pred in segment_predictions:
        key = (
            str(pred.get("phone") or "").upper(),
            round(float(pred.get("start") or 0.0), 2),
        )
        pred_lookup[key] = pred

    results: list[dict[str, Any]] = []
    detail_iter = iter(details)

    for seg in alignment_phones:
        detail = next(detail_iter, None)
        if detail is None:
            break
        if detail.get("mapping") == "silence":
            continue

        canonical = detail.get("canonical")
        if not canonical:
            # Do not expose raw MFA phones when canonicalization is unsafe.
            continue
        start = seg.get("start")
        end = seg.get("end")

        # Try to find a matching prediction by (raw_phone_upper, rounded_start)
        raw_key = (str(seg.get("phone") or "").upper(), round(float(start or 0.0), 2))
        prediction = pred_lookup.get(raw_key)

        if prediction is None:
            # Fallback: match by canonical phone
            for pred in segment_predictions:
                if str(pred.get("phone") or "") == canonical:
                    prediction = pred
                    break

        if prediction:
            error_type = prediction.get("predicted_error_type")
            confidence = prediction.get("diagnosis_confidence")
            is_issue = (
                error_type not in {None, "unknown"}
                and float(confidence or 0.0) >= 0.45
            )
        else:
            error_type = None
            confidence = None
            is_issue = False

        if reliability_status == "invalid":
            is_issue = False
            error_type = None
            confidence = None

        results.append({
            "phone": canonical,
            "status": "needs_improvement" if is_issue else ("unknown" if prediction is None else "ok"),
            "error_type": error_type if is_issue else None,
            "start_time": round(float(start), 3) if start is not None else None,
            "end_time": round(float(end), 3) if end is not None else None,
            "diagnosis_confidence": round(float(confidence), 3) if confidence is not None and is_issue else None,
        })

    return results
