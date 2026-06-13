from continual_learning.text_utils import (
    answer_matches,
    approx_tokens,
    chunk_words,
    parse_json_block,
)


class TestApproxTokens:
    def test_empty_string(self):
        assert approx_tokens("") == 0

    def test_word_ratio(self):
        # 3 words / 0.75 = 4
        assert approx_tokens("one two three") == 4


class TestChunkWords:
    def test_short_text_single_chunk(self):
        assert chunk_words("a b c", 10) == ["a b c"]

    def test_strips_whitespace(self):
        assert chunk_words("  a b  ", 10) == ["a b"]

    def test_empty_text_returns_empty(self):
        assert chunk_words("   ", 10) == []

    def test_splits_into_fixed_chunks(self):
        text = " ".join(str(i) for i in range(10))
        chunks = chunk_words(text, 4)
        assert chunks == ["0 1 2 3", "4 5 6 7", "8 9"]


class TestParseJsonBlock:
    def test_plain_json(self):
        assert parse_json_block('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        raw = '```json\n{"a": 1}\n```'
        assert parse_json_block(raw) == {"a": 1}

    def test_invalid_returns_none(self):
        assert parse_json_block("not json") is None


class TestAnswerMatches:
    def test_case_insensitive_match(self):
        assert answer_matches("Oracle", "the ORACLE database") is True

    def test_no_match(self):
        assert answer_matches("Postgres", "Oracle database") is False
