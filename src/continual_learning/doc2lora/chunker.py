class DocumentChunker:
    """Splits documents into fixed-size token chunks for hypernetwork processing.

    In 'doc' mode, splits text into chunks of `chunk_size` words.
    In 'text' mode (task descriptions), returns the full text as a single chunk.
    """

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    def chunk(self, text: str, mode: str = "doc") -> list[str]:
        text = text.strip()
        if not text:
            return []

        if mode == "text":
            return [text]

        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks
