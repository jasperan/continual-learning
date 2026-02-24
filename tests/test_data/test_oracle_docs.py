import pytest
from continual_learning.data.oracle_docs import chunk_text, parse_html_to_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_multiple_chunks(self):
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1

    def test_chunks_respect_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_tokens=10)
        for chunk in chunks:
            assert chunk.endswith(".") or chunk == chunks[-1]

    def test_no_empty_chunks(self):
        text = "Some text here. And more text."
        chunks = chunk_text(text, max_tokens=5)
        for chunk in chunks:
            assert len(chunk.strip()) > 0


class TestParseHtmlToText:
    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        text = parse_html_to_text(html)
        assert "<p>" not in text
        assert "<b>" not in text
        assert "Hello" in text
        assert "world" in text

    def test_handles_plain_text(self):
        text = parse_html_to_text("Just plain text.")
        assert text == "Just plain text."
