"""Lexical, hybrid and graph retrieval: BM25, RRF fusion, reranking, entities.

- ``BM25Index``: Okapi BM25 over an in-memory passage collection.
- ``reciprocal_rank_fusion``: rank fusion across retrieval runs (Cormack et al.).
- ``borda_count_fusion``: Borda count rank aggregation across multiple rankings.
- ``HybridRetriever``: BM25 plus any dense index with ``search(vector, k)``
  (e.g. ``ox.VectorIndex``) fused by RRF.
- ``CrossEncoderReranker``: exact reranking with a caller-provided scorer.
- ``EntityGraph`` / ``summarize_communities``: co-occurrence entity graph with
  deterministic label-propagation communities (GraphRAG-lite).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import re

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall(str(text).lower())


class BM25Index:
    """Okapi BM25 retrieval over an in-memory document collection."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("need k1 >= 0 and b within [0, 1]")
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, Dict[str, int]] = {}
        self._df: Dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        """Clears all indexed documents and vocabulary statistics."""
        self._docs.clear()
        self._df.clear()
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        """Indexes a document; duplicate ids are refused."""
        doc_id = str(doc_id)
        if doc_id in self._docs:
            raise ValueError(f"duplicate doc_id '{doc_id}'")
        counts: Dict[str, int] = {}
        for tok in _tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        self._docs[doc_id] = counts
        for tok in counts:
            self._df[tok] = self._df.get(tok, 0) + 1
        self._total_len += sum(counts.values())

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Top-k (id, score) pairs, highest first; empty when nothing matches."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._docs:
            return []
        avgdl = self._total_len / len(self._docs)
        q_terms = _tokens(query)
        if not q_terms:
            return []
        scored = []
        for doc_id, counts in self._docs.items():
            dl = sum(counts.values())
            total = 0.0
            for term in set(q_terms):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                total += self._idf(term) * tf * (self.k1 + 1.0) / denom
            if total > 0:
                scored.append((doc_id, total))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]], k: float = 60.0
) -> List[Tuple[Any, float]]:
    """Fuses best-first id rankings by sum of 1/(k + rank)."""
    if k <= 0:
        raise ValueError("k must be positive")
    fused: Dict[Any, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], str(pair[0])))


def borda_count_fusion(
    rankings: Sequence[Sequence[Any]],
) -> List[Tuple[Any, float]]:
    """Fuses rankings using the Borda count voting rule."""
    scores: Dict[Any, float] = {}
    for ranking in rankings:
        n = len(ranking)
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            points = float(n - rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + points
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def _hit_id(hit: Any) -> Any:
    if isinstance(hit, (tuple, list)):
        return hit[0]
    hit_id = getattr(hit, "id", None)
    if hit_id is None:
        raise ValueError(f"dense hit {hit!r} carries no id")
    return hit_id


class HybridRetriever:
    """BM25 plus dense-index retrieval fused with reciprocal rank fusion."""

    def __init__(
        self,
        lexical: BM25Index,
        dense: Any,
        embed: Callable[[str], Any],
        rrf_k: float = 60.0,
    ):
        if not isinstance(lexical, BM25Index):
            raise ValueError("lexical must be a BM25Index")
        if not callable(getattr(dense, "search", None)):
            raise ValueError("dense must provide search(vector, k)")
        if not callable(embed):
            raise TypeError("embed must be callable")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.lexical = lexical
        self.dense = dense
        self.embed = embed
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 5,
        lexical_k: Optional[int] = None,
        dense_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Fused top-k (id, score) pairs across both retrievers."""
        if k < 1:
            raise ValueError("k must be >= 1")
        lex_k = k if lexical_k is None else lexical_k
        den_k = k if dense_k is None else dense_k
        if lex_k < 1 or den_k < 1:
            raise ValueError("lexical_k and dense_k must be >= 1")
        lex_ranking = [doc_id for doc_id, _ in self.lexical.search(query, lex_k)]
        dense_ranking = [_hit_id(hit) for hit in self.dense.search(self.embed(query), den_k)]
        return reciprocal_rank_fusion([lex_ranking, dense_ranking], self.rrf_k)[:k]


class CrossEncoderReranker:
    """Exact reranking with a caller-provided (query, doc) -> score function."""

    def __init__(self, scorer: Callable[[str, str], float]):
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer

    def rerank(
        self, query: str, docs: Sequence[str], top_n: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """(index, score, doc) triples, highest score first."""
        docs = list(docs)
        if not docs or not all(isinstance(d, str) for d in docs):
            raise ValueError("docs must be a non-empty string list")
        n = len(docs) if top_n is None else top_n
        if not 1 <= n <= len(docs):
            raise ValueError(f"top_n = {top_n} outside [1, {len(docs)}]")
        scored = []
        for i, doc in enumerate(docs):
            score = float(self.scorer(query, doc))
            if not math.isfinite(score):
                raise ValueError(f"scorer returned non-finite score for doc {i}")
            scored.append((i, score, doc))
        scored.sort(key=lambda triple: (-triple[1], triple[0]))
        return scored[:n]


class EntityGraph:
    """Undirected entity co-occurrence graph with passage provenance."""

    def __init__(self):
        self._adj: Dict[str, Dict[str, float]] = {}
        self._texts: Dict[str, str] = {}
        self._members: Dict[str, List[str]] = {}

    def add_passage(self, passage_id: str, text: str, entities: Sequence[str]) -> None:
        """Adds a passage and links its (deduped) entities pairwise."""
        passage_id = str(passage_id)
        if passage_id in self._texts:
            raise ValueError(f"duplicate passage_id '{passage_id}'")
        cleaned = [e.strip() for e in entities if e.strip()]
        ordered = list(dict.fromkeys(cleaned))
        self._texts[passage_id] = str(text)
        self._members[passage_id] = ordered
        for entity in ordered:
            self._adj.setdefault(entity, {})
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                self._adj[left][right] = self._adj[left].get(right, 0.0) + 1.0
                self._adj[right][left] = self._adj[right].get(left, 0.0) + 1.0

    def nodes(self) -> List[str]:
        return sorted(self._adj)

    def edges(self) -> List[Tuple[str, str, float]]:
        seen = set()
        out = []
        for left in sorted(self._adj):
            for right in sorted(self._adj[left]):
                if (right, left) in seen:
                    continue
                seen.add((left, right))
                out.append((left, right, self._adj[left][right]))
        return out

    def passages(self, entity: str) -> List[str]:
        """Sorted passage ids mentioning the entity."""
        return sorted(pid for pid, ents in self._members.items() if entity in ents)

    def degree_centrality(self) -> Dict[str, float]:
        """Computes degree centrality scores for all entity nodes."""
        n = len(self._adj)
        if n <= 1:
            return {node: 0.0 for node in self._adj}
        denom = float(n - 1)
        return {node: len(neighbors) / denom for node, neighbors in self._adj.items()}

    def communities(self, max_iter: int = 20) -> List[List[str]]:
        """Deterministic label-propagation communities (async, sorted order)."""
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        labels = {node: node for node in self._adj}
        for _ in range(max_iter):
            moved = False
            for node in sorted(labels):
                weights: Dict[str, float] = {}
                for other, weight in self._adj[node].items():
                    label = labels[other]
                    weights[label] = weights.get(label, 0.0) + weight
                if not weights:
                    continue
                best = sorted(weights, key=lambda lab: (-weights[lab], lab))[0]
                if best != labels[node]:
                    labels[node] = best
                    moved = True
            if not moved:
                break
        buckets: Dict[str, List[str]] = {}
        for node, label in labels.items():
            buckets.setdefault(label, []).append(node)
        return sorted([sorted(members) for members in buckets.values()])


def summarize_communities(
    graph: EntityGraph,
    communities: Sequence[Sequence[str]],
    summarize: Callable[[List[str], List[str]], str],
) -> Dict[int, str]:
    """Summarizes each community from its entities and passage texts."""
    if not callable(summarize):
        raise TypeError("summarize must be callable")
    out = {}
    for idx, community in enumerate(communities):
        entities = sorted(set(community))
        if not entities:
            raise ValueError(f"community {idx} is empty")
        unknown = [e for e in entities if e not in graph.nodes()]
        if unknown:
            raise ValueError(f"community {idx} holds unknown entities {unknown}")
        seen: List[str] = []
        for entity in entities:
            for pid in graph.passages(entity):
                if pid not in seen:
                    seen.append(pid)
        texts = [graph._texts[pid] for pid in sorted(seen)]
        out[idx] = summarize(entities, texts)
    return out


__all__ = [
    "BM25Index",
    "reciprocal_rank_fusion",
    "borda_count_fusion",
    "HybridRetriever",
    "CrossEncoderReranker",
    "EntityGraph",
    "summarize_communities",
]
