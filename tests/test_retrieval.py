"""Tests for retrieval (oxideon.retrieval)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    BM25Index,
    CrossEncoderReranker,
    EntityGraph,
    HybridRetriever,
    VectorIndex,
    reciprocal_rank_fusion,
    summarize_communities,
)


def test_top_level_exports():
    assert ox.BM25Index is BM25Index
    assert ox.HybridRetriever is HybridRetriever
    assert ox.retrieval.EntityGraph is EntityGraph


def test_bm25_hand_computed_ranking():
    idx = BM25Index()
    idx.add("d1", "cat cat dog")
    idx.add("d2", "bird")
    assert len(idx) == 2
    hits = idx.search("cat")
    assert [doc for doc, _ in hits] == ["d1"]
    assert hits[0][1] == pytest.approx(0.8531, abs=1e-4)
    assert idx.search("cat", k=5)[0][0] == "d1"
    assert idx.search("zebra") == []
    assert idx.search("") == []
    with pytest.raises(ValueError):
        idx.add("d1", "again")
    with pytest.raises(ValueError):
        idx.search("cat", k=0)
    assert BM25Index().search("cat") == []


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [doc for doc, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], ["x"]]) == [("x", pytest.approx(1 / 61))]
    assert [d for d, _ in reciprocal_rank_fusion([["a", "a", "b"]])] == ["a", "b"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0.0)


def test_hybrid_retriever_fuses_lexical_and_dense():
    lex = BM25Index()
    lex.add("d1", "apple fruit")
    lex.add("d2", "car engine")
    dense = VectorIndex(dim=2)
    dense.add([[1.0, 0.0], [0.0, 1.0]], ids=["d1", "d2"])
    hybrid = HybridRetriever(lex, dense, lambda q: [1.0, 0.0])
    hits = hybrid.search("apple", k=2)
    assert hits[0][0] == "d1"
    hits = hybrid.search("engine", k=2)
    assert [doc for doc, _ in hits] == ["d2", "d1"]
    assert hits[0][1] > hits[1][1]
    with pytest.raises(ValueError):
        hybrid.search("apple", k=0)
    with pytest.raises(TypeError):
        HybridRetriever(lex, dense, "not-callable")
    with pytest.raises(ValueError):
        HybridRetriever(lex, object(), lambda q: [1.0, 0.0])


def test_cross_encoder_reranker():
    ranker = CrossEncoderReranker(lambda q, d: float(len(d)))
    out = ranker.rerank("q", ["a", "abcd", "ab"])
    assert [(i, s) for i, s, _ in out] == [(1, 4.0), (2, 2.0), (0, 1.0)]
    assert len(ranker.rerank("q", ["a", "abcd", "ab"], top_n=2)) == 2
    with pytest.raises(ValueError):
        ranker.rerank("q", [])
    with pytest.raises(ValueError):
        ranker.rerank("q", ["a"], top_n=2)
    with pytest.raises(ValueError):
        CrossEncoderReranker(lambda q, d: float("nan")).rerank("q", ["a"])
    with pytest.raises(TypeError):
        CrossEncoderReranker(None)


def test_entity_graph_communities_and_provenance():
    graph = EntityGraph()
    graph.add_passage("p1", "Alice met Bob", ["Alice", "Bob"])
    graph.add_passage("p2", "Bob saw Carol", ["Bob", "Carol"])
    graph.add_passage("p3", "Zed alone", ["Zed"])
    assert graph.nodes() == ["Alice", "Bob", "Carol", "Zed"]
    assert graph.edges() == [("Alice", "Bob", 1.0), ("Bob", "Carol", 1.0)]
    assert graph.passages("Bob") == ["p1", "p2"]
    assert graph.communities() == [["Alice", "Bob", "Carol"], ["Zed"]]
    out = summarize_communities(
        graph, graph.communities(), lambda ents, texts: f"{len(ents)}:{len(texts)}"
    )
    assert out == {0: "3:2", 1: "1:1"}
    with pytest.raises(ValueError):
        graph.add_passage("p1", "dup", ["X"])
    with pytest.raises(ValueError):
        graph.communities(max_iter=0)
    with pytest.raises(ValueError):
        summarize_communities(graph, [["Ghost"]], lambda e, t: "")
    with pytest.raises(ValueError):
        summarize_communities(graph, [[]], lambda e, t: "")
    with pytest.raises(TypeError):
        summarize_communities(graph, [["Zed"]], None)
    assert EntityGraph().communities() == []
