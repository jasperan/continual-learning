from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from continual_learning.text_utils import chunk_words


class TFIDFRetriever:
    """TF-IDF based document chunk retriever."""

    def __init__(self):
        self._chunks: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def add_document(self, text: str, max_chunk_words: int = 300) -> int:
        new_chunks = chunk_words(text, max_chunk_words)
        self._chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self._chunks or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_indices if scores[i] > 0]

    def clear(self) -> None:
        self._chunks.clear()
        self._vectorizer = None
        self._matrix = None

    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(self._chunks)
