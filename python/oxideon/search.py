"""In-memory Vector Database and Semantic Nearest-Neighbor Search Index for Oxideon."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import json
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import softmax


class SearchResult(NamedTuple):
    id: Union[int, str]
    score: float
    metadata: Optional[Dict[str, Any]] = None


class VectorIndex:
    """Fast in-memory Vector Database for embedding retrieval and semantic search."""

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = int(dim)
        self.metric = metric.lower()
        if self.metric not in ("cosine", "l2", "dot"):
            raise ValueError(f"Unsupported metric '{metric}'. Expected 'cosine', 'l2', or 'dot'.")

        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.ids: List[Union[int, str]] = []
        self.metadatas: List[Optional[Dict[str, Any]]] = []

    def clear(self) -> None:
        """Clears all vectors, IDs, and metadata from the index."""
        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.ids = []
        self.metadatas = []

    def add(
        self,
        vectors: Union[np.ndarray, _Tensor, Sequence[Sequence[float]]],
        ids: Optional[Sequence[Union[int, str]]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ):
        """Adds a collection of vectors and optional metadata to the index."""
        if hasattr(vectors, "to_numpy"):
            arr = vectors.to_numpy().astype(np.float32)
        else:
            arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        assert arr.shape[1] == self.dim, f"Dimension mismatch: expected {self.dim}, got {arr.shape[1]}"

        n = arr.shape[0]
        if ids is None:
            start_id = len(self.ids)
            ids = list(range(start_id, start_id + n))
        else:
            ids = list(ids)
            if len(ids) != n:
                raise ValueError(f"ids length mismatch: expected {n}, got {len(ids)}")

        if metadatas is None:
            metadatas = [None] * n
        else:
            metadatas = list(metadatas)
            if len(metadatas) != n:
                raise ValueError(f"metadatas length mismatch: expected {n}, got {len(metadatas)}")

        self.vectors = np.vstack([self.vectors, arr]) if self.vectors.size > 0 else arr
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(
        self,
        query: Union[np.ndarray, _Tensor, Sequence[float]],
        k: int = 5,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[SearchResult]:
        """Searches for the top-k most similar vectors."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if len(self.ids) == 0:
            return []

        if hasattr(query, "to_numpy"):
            q = query.to_numpy().astype(np.float32).reshape(-1)
        else:
            q = np.asarray(query, dtype=np.float32).reshape(-1)

        assert q.shape[0] == self.dim, f"Query dim mismatch: expected {self.dim}, got {q.shape[0]}"

        if self.metric == "cosine":
            q_norm = np.linalg.norm(q) + 1e-12
            v_norm = np.linalg.norm(self.vectors, axis=1) + 1e-12
            scores = np.dot(self.vectors, q) / (v_norm * q_norm)
        elif self.metric == "dot":
            scores = np.dot(self.vectors, q)
        elif self.metric == "l2":
            dists = np.linalg.norm(self.vectors - q, axis=1)
            scores = -dists  # Negate so higher is closer

        if filter_fn is not None:
            valid_indices = [
                i for i, m in enumerate(self.metadatas)
                if m is not None and filter_fn(m)
            ]
            if not valid_indices:
                return []
            valid_indices = np.array(valid_indices, dtype=np.int64)
            filtered_scores = scores[valid_indices]
            top_k_sub = np.argsort(-filtered_scores)[:k]
            top_indices = valid_indices[top_k_sub]
        else:
            top_indices = np.argsort(-scores)[:k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                id=self.ids[idx],
                score=float(scores[idx]),
                metadata=self.metadatas[idx],
            ))
        return results

    def search_batch(
        self,
        queries: Union[np.ndarray, _Tensor, Sequence[Sequence[float]]],
        k: int = 5,
    ) -> List[List[SearchResult]]:
        """Searches top-k similar items for a batch of query vectors."""
        if hasattr(queries, "to_numpy"):
            q_arr = queries.to_numpy().astype(np.float32)
        else:
            q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)

        return [self.search(q, k=k) for q in q_arr]

    def save(self, filepath: str):
        """Serializes index vectors and metadata."""
        data = {
            "dim": self.dim,
            "metric": self.metric,
            "vectors": self.vectors.tolist(),
            "ids": self.ids,
            "metadatas": self.metadatas,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "VectorIndex":
        """Loads a persisted VectorIndex."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(dim=data["dim"], metric=data["metric"])
        idx.vectors = np.array(data["vectors"], dtype=np.float32)
        idx.ids = data["ids"]
        idx.metadatas = data["metadatas"]
        return idx

    def __len__(self) -> int:
        return len(self.ids)


def differentiable_knn(
    query: _Tensor,
    index_vectors: _Tensor,
    k: int = 5,
    temperature: float = 1.0,
) -> Tuple[_Tensor, _Tensor]:
    """Differentiable continuous k-NN retrieval via soft temperature attention."""
    temp = max(float(temperature), 1e-4)
    # query: [B, D], index_vectors: [N, D]
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if index_vectors.ndim == 1:
        index_vectors = index_vectors.unsqueeze(0)

    # Cosine or scaled dot product similarity: [B, N]
    sim = query.matmul(index_vectors.t())
    scaled_sim = sim * (1.0 / temp)
    soft_weights = softmax(scaled_sim, dim=-1)

    # Soft retrieved representation: [B, D]
    retrieved = soft_weights.matmul(index_vectors)
    return soft_weights, retrieved


__all__ = [
    "SearchResult",
    "VectorIndex",
    "differentiable_knn",
]
