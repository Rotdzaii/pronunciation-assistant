from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.contracts.alignment_contract import AlignmentError


_WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class TranscriptNormalizationResult:
    text: str
    words: list[str]
    oov_words: list[str]
    dictionary_checked: bool


def normalize_transcript(value: str | None) -> tuple[str, list[str]]:
    """Return an MFA-safe transcript while preserving dictionary lookup words."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    words = _WORD_RE.findall(normalized)
    if not words:
        raise AlignmentError(
            "Transcript is empty or contains no dictionary-compatible words after normalization.",
            code="empty_transcript",
        )
    return " ".join(words), words


def dictionary_words(dictionary_path: str | Path | None) -> set[str] | None:
    """Read a plain MFA dictionary when a local file is supplied.

    MFA model registry names are intentionally not guessed here: their vocabulary
    is owned by MFA and must be reported by MFA itself on an OOV failure.
    """

    if not dictionary_path:
        return None
    path = Path(dictionary_path).expanduser()
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AlignmentError("Unable to read the configured MFA dictionary.", code="dictionary_invalid") from exc

    vocabulary: set[str] = set()
    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith(("#", ";")):
            continue
        word = candidate.split()[0].split("(")[0]
        if word:
            vocabulary.add(word.casefold())
    return vocabulary


def normalize_and_check_transcript(
    value: str | None,
    dictionary_path: str | Path | None = None,
) -> TranscriptNormalizationResult:
    text, words = normalize_transcript(value)
    vocabulary = dictionary_words(dictionary_path)
    oov_words = [] if vocabulary is None else sorted({word for word in words if word.casefold() not in vocabulary})
    return TranscriptNormalizationResult(
        text=text,
        words=words,
        oov_words=oov_words,
        dictionary_checked=vocabulary is not None,
    )
