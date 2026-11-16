"""Retrieval-Augmented Generation (RAG) pipeline infrastructure for Oxideon.

Connects vector indexing, passage retrieval, context compression, hybrid Reciprocal Rank Fusion (RRF), and prompt templating:
  - Document: Structured text chunk holding metadata, content, and identifiers
  - RAGRetriever: High-throughput vector retrieval engine querying documents
  - PromptTemplate: Parametric prompt formatting for grounded generation
  - ContextCompressor: Truncates, deduplicates, and formats retrieved passages
  - reciprocal_rank_fusion: Combines multiple ranked document lists via RRF (Cormack et al., 2009)
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence
import re
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .search import VectorIndex


class Document:
    """A passage or document chunk for retrieval-augmented generation."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        self.content = str(content)
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(hash(self.content))

    def __repr__(self) -> str:
        snippet = self.content[:40] + ("..." if len(self.content) > 40 else "")
        return f"Document(id={self.doc_id}, content='{snippet}')"


class PromptTemplate:
    """Parametric prompt template with named slot substitutions (respecting escaped {{braces}})."""

    def __init__(self, template: str):
        self.template = template
        # Strip escaped double braces before extracting {slot} names
        unescaped = template.replace("{{", "").replace("}}", "")
        self.slots = list(dict.fromkeys(re.findall(r"\{([a-zA-Z0-9_]+)\}", unescaped)))

    def format(self, **kwargs: Any) -> str:
        missing = [s for s in self.slots if s not in kwargs]
        if missing:
            raise KeyError(f"Missing required template slot(s): {missing}")
        return self.template.format(**kwargs)

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Partially binds a subset of template slots, returning a new PromptTemplate."""
        new_tpl = self.template
        for k, v in kwargs.items():
            # Replace single-brace slot {k} without clobbering escaped {{k}}
            pattern = r"(?<!\{)\{" + re.escape(str(k)) + r"\}(?!\})"
            new_tpl = re.sub(pattern, lambda _m, val=str(v): val, new_tpl)
        return PromptTemplate(new_tpl)

    def __repr__(self) -> str:
        return f"PromptTemplate(slots={self.slots})"


class ContextCompressor:
    """Compresses and formats multiple retrieved documents into a clean prompt context block."""

    def __init__(self, max_chars: int = 2000, separator: str = "\n---\n"):
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        self.max_chars = max_chars
        self.separator = separator

    def compress(self, documents: List[Document]) -> str:
        """Formats and deduplicates document chunks, respecting max_chars."""
        seen = set()
        chunks = []
        curr_len = 0

        for doc in documents:
            text = doc.content.strip()
            if not text or text in seen:
                continue
            seen.add(text)

            chunk = f"[{doc.doc_id}] {text}"
            sep_len = len(self.separator) if chunks else 0
            if curr_len + sep_len + len(chunk) > self.max_chars:
                if not chunks:
                    chunks.append(chunk[: self.max_chars])
                break
            chunks.append(chunk)
            curr_len += sep_len + len(chunk)

        return self.separator.join(chunks)


class RAGRetriever:
    """Vector-backed passage retrieval engine."""

    def __init__(self, vector_index: VectorIndex, documents: List[Document]):
        self.index = vector_index
        self.documents = list(documents)
        if len(self.index) != len(self.documents):
            raise ValueError(
                f"Index size ({len(self.index)}) must match document count ({len(self.documents)})"
            )

    def retrieve(self, query_embedding: Any, k: int = 3) -> List[Tuple[Document, float]]:
        """Queries the vector index and returns top-k matching (Document, score) pairs."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        q_tensor = (
            query_embedding
            if isinstance(query_embedding, _Tensor)
            else _from_numpy(np.ascontiguousarray(query_embedding, dtype=np.float32))
        )
        search_results = self.index.search(q_tensor, k=k)
        doc_map = {doc.doc_id: doc for doc in self.documents}
        results = []
        for r in search_results:
            r_str = str(r.id)
            if r_str in doc_map:
                results.append((doc_map[r_str], float(r.score)))
            elif isinstance(r.id, (int, np.integer)) and 0 <= int(r.id) < len(self.documents):
                results.append((self.documents[int(r.id)], float(r.score)))
            else:
                for doc in self.documents:
                    if doc.doc_id == r_str:
                        results.append((doc, float(r.score)))
                        break
        return results


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Any]], k: int = 60
) -> List[Tuple[Document, float]]:
    """Combines multiple ranked retrieval lists via Reciprocal Rank Fusion (RRF).

    Each item in a ranked list may be either a `Document` or a `(Document, score)` tuple.
    Returns `(Document, rrf_score)` sorted by descending RRF score.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for r_list in ranked_lists:
        for rank, item in enumerate(r_list, start=1):
            doc = item[0] if isinstance(item, tuple) else item
            did = str(doc.doc_id)
            docs[did] = doc
            scores[did] = scores.get(did, 0.0) + 1.0 / float(k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(docs[did], sc) for did, sc in ordered]


__all__ = [
    "Document",
    "PromptTemplate",
    "ContextCompressor",
    "RAGRetriever",
    "reciprocal_rank_fusion",
]
