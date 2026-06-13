"""Shared text helpers used across strategies.

Small pure functions that were previously hand-rolled in multiple engines:
token estimation, fixed-size word chunking, fenced-JSON parsing, and
substring-based QA scoring.
"""

import json

# Approximate words-per-token ratio (1 token ~ 0.75 words).
_WORDS_PER_TOKEN = 0.75


def approx_tokens(text: str) -> int:
    """Estimate token count from word count (1 token ~ 0.75 words)."""
    return int(len(text.split()) / _WORDS_PER_TOKEN)


def chunk_words(text: str, max_words: int) -> list[str]:
    """Split ``text`` into fixed-size word chunks of at most ``max_words`` words.

    Whitespace-only chunks are dropped. Returns a single-element list with the
    stripped text when it fits in one chunk, matching the legacy chunkers.
    """
    words = text.split()
    if len(words) <= max_words:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def parse_json_block(raw: str):
    """Parse a JSON object, tolerating a leading/trailing markdown code fence.

    Returns the decoded object, or ``None`` if it cannot be parsed.
    """
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def answer_matches(answer: str, response: str) -> bool:
    """True if the expected ``answer`` appears (case-insensitively) in ``response``."""
    return answer.lower() in response.lower()
