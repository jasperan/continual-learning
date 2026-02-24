import torch
import torch.nn.functional as F


class KnowledgeStore:
    """Stores document knowledge as mean-pooled hidden-state embeddings."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self._doc_names: list[str] = []
        self._doc_embeddings: list[torch.Tensor] = []
        self._doc_token_embeddings: list[torch.Tensor] = []

    @property
    def num_entries(self) -> int:
        return len(self._doc_names)

    def add(self, name: str, token_embeddings: torch.Tensor) -> None:
        self._doc_names.append(name)
        self._doc_token_embeddings.append(token_embeddings.detach().cpu())
        self._doc_embeddings.append(token_embeddings.detach().cpu().mean(dim=0))

    def query(self, query_embedding: torch.Tensor, top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._doc_embeddings:
            return torch.zeros(0), torch.zeros(0, self.hidden_size)

        query_flat = query_embedding.detach().cpu().flatten()
        doc_matrix = torch.stack(self._doc_embeddings)

        scores = F.cosine_similarity(query_flat.unsqueeze(0), doc_matrix)

        k = min(top_k, len(self._doc_embeddings))
        top_scores, top_indices = scores.topk(k)
        top_embeddings = doc_matrix[top_indices]

        return top_scores, top_embeddings

    def get_token_embeddings(self, index: int) -> torch.Tensor:
        return self._doc_token_embeddings[index]

    def clear(self) -> None:
        self._doc_names.clear()
        self._doc_embeddings.clear()
        self._doc_token_embeddings.clear()
