from continual_learning.doc2lora.chunker import DocumentChunker


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=10)

    def test_short_text_single_chunk(self):
        chunks = self.chunker.chunk("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_exact_chunk_size(self):
        text = " ".join(["word"] * 10)
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        text = " ".join(["word"] * 25)
        chunker = DocumentChunker(chunk_size=10)
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_empty_text(self):
        chunks = self.chunker.chunk("")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunks = self.chunker.chunk("   \n\n  ")
        assert len(chunks) == 0

    def test_default_chunk_size(self):
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1024

    def test_text_mode_no_chunking(self):
        text = " ".join(["word"] * 100)
        chunks = self.chunker.chunk(text, mode="text")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_doc_mode_chunks(self):
        text = " ".join(["word"] * 25)
        chunker = DocumentChunker(chunk_size=10)
        chunks = chunker.chunk(text, mode="doc")
        assert len(chunks) == 3
