"""Comprehensive tests for Oxideon vectorization suite: Count, TF-IDF, Hashing, BM25, Word2Vec, Spectral, and Contextual Pooling."""

import numpy as np
import pytest

from oxideon import (
    BM25Vectorizer,
    ByteHistogramVectorizer,
    ColBERTMaxSimVectorizer,
    ContextualPoolingVectorizer,
    CountVectorizer,
    DeepSetsVectorizer,
    DictVectorizer,
    DiscretizationVectorizer,
    Doc2VecDBOWVectorizer,
    ENGLISH_STOP_WORDS,
    FastBM25Index,
    FastByteHistogramVectorizer,
    FastCountVectorizer,
    FastDiscretizer,
    FastHashingVectorizer,
    FastLsaVectorizer,
    FastMaxSimScorer,
    FastMinHashVectorizer,
    FastSimHashVectorizer,
    FastSphericalHarmonics,
    FastTextCharNGramVectorizer,
    FastTfidfVectorizer,
    HashingVectorizer,
    JsonTreeVectorizer,
    KernelSpectralVectorizer,
    OneHotVectorizer,
    PathSignatureVectorizer,
    PositionalSpectralVectorizer,
    SphericalHarmonicsVectorizer,
    SubwordTfidfVectorizer,
    Tensor,
    TfidfTransformer,
    TfidfVectorizer,
    TimeSeriesSpectralVectorizer,
    TopicModelVectorizer,
    WeisfeilerLehmanVectorizer,
    Word2VecVectorizer,
    fast_murmurhash3_32,
    from_numpy,
)
from oxideon.sparse import CSRTensor


CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Continuous representations of words and phrases and their compositionality.",
    "Deep residual learning for image recognition and natural language processing.",
    "Attention is all you need for neural sequence transduction models.",
]


def test_count_vectorizer_basic_and_ngrams():
    # 1. Basic unigram count vectorizer
    vec = CountVectorizer(lowercase=True)
    X = vec.fit_transform(CORPUS)
    assert isinstance(X, Tensor)
    assert X.shape[0] == 4
    assert X.shape[1] > 20

    # Test feature names
    features = vec.get_feature_names_out()
    assert "fox" in features
    assert "dog" in features
    assert "attention" in features

    # Verify counts for document 0
    fox_idx = vec.vocabulary_["fox"]
    doc0_counts = X.to_numpy()[0]
    assert doc0_counts[fox_idx] == 1.0

    # 2. Bigrams (1, 2)
    vec_bigram = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    X_bi = vec_bigram.fit_transform(CORPUS)
    assert X_bi.shape[1] > X.shape[1]
    assert "quick brown" in vec_bigram.vocabulary_
    assert "brown fox" in vec_bigram.vocabulary_

    # 3. Sparse CSR output
    X_sparse = vec.transform(CORPUS, return_sparse=True)
    assert isinstance(X_sparse, CSRTensor)
    assert X_sparse.shape == X.shape
    np.testing.assert_allclose(X_sparse.to_dense().to_numpy(), X.to_numpy())

    # 4. Inverse transform
    inv = vec.inverse_transform(X)
    assert len(inv) == 4
    assert "fox" in inv[0]


def test_tfidf_transformer_and_vectorizer():
    # 1. TfidfTransformer on count matrix
    cv = CountVectorizer(stop_words="english")
    counts = cv.fit_transform(CORPUS)

    tfidf_trans = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=True)
    X_tfidf = tfidf_trans.fit_transform(counts)
    assert isinstance(X_tfidf, Tensor)
    assert X_tfidf.shape == counts.shape

    # L2 norm check
    arr = X_tfidf.to_numpy()
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5, atol=1e-5)

    # 2. TfidfVectorizer end-to-end
    tvec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", norm="l2")
    X_out = tvec.fit_transform(CORPUS)
    assert X_out.shape[0] == 4
    out_arr = X_out.to_numpy()
    out_norms = np.linalg.norm(out_arr, axis=1)
    np.testing.assert_allclose(out_norms, np.ones(4), rtol=1e-5, atol=1e-5)


def test_hashing_vectorizer():
    # Stateless hashing vectorizer
    hvec = HashingVectorizer(n_features=512, alternate_sign=True, norm="l2")
    X_hash = hvec.transform(CORPUS)
    assert X_hash.shape == (4, 512)

    # L2 norm check
    arr = X_hash.to_numpy()
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5, atol=1e-5)

    # Deterministic test
    X_hash2 = hvec.transform(CORPUS)
    np.testing.assert_allclose(X_hash.to_numpy(), X_hash2.to_numpy())

    # Sparse return
    X_sparse = hvec.transform(CORPUS, return_sparse=True)
    assert isinstance(X_sparse, CSRTensor)
    assert X_sparse.shape == (4, 512)


def test_bm25_vectorizer_and_retrieval():
    bm25 = BM25Vectorizer(k1=1.5, b=0.75, delta=0.25, stop_words="english")
    doc_vectors = bm25.fit_transform(CORPUS)
    assert doc_vectors.shape[0] == 4
    assert len(bm25.vocabulary_) > 10

    # Query: "neural sequence models"
    ranked = bm25.score_query("neural sequence models", doc_vectors, top_k=2)
    top_doc_idx, top_score = ranked[0]
    # Document 3 is "Attention is all you need for neural sequence transduction models."
    assert top_doc_idx == 3
    assert top_score > 0.0

    # Query: "brown fox dog"
    ranked_fox = bm25.score_query("brown fox dog", doc_vectors, top_k=1)
    assert ranked_fox[0][0] == 0


def test_word2vec_vectorizer_mean_and_sif():
    # Fit PPMI + SVD semantic representations
    w2v = Word2VecVectorizer(embedding_dim=32, pooling="mean", normalize=True)
    X_dense = w2v.fit_transform(CORPUS)
    assert X_dense.shape == (4, 32)
    norms = np.linalg.norm(X_dense.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5, atol=1e-5)

    # Test SIF pooling
    w2v_sif = Word2VecVectorizer(embedding_dim=32, pooling="sif", normalize=True)
    X_sif = w2v_sif.fit_transform(CORPUS)
    assert X_sif.shape == (4, 32)
    sif_norms = np.linalg.norm(X_sif.to_numpy(), axis=1)
    np.testing.assert_allclose(sif_norms, np.ones(4), rtol=1e-5, atol=1e-5)


def test_positional_spectral_vectorizer():
    spec_vec = PositionalSpectralVectorizer(num_frequencies=8, max_seq_len=128)
    assert spec_vec.dim == 16  # 8 * 2

    positions = np.array([0, 10, 32, 64], dtype=np.float32)
    encoded = spec_vec.encode_positions(positions)
    assert encoded.shape == (4, 16)
    assert not np.isnan(encoded.to_numpy()).any()


def test_contextual_pooling_vectorizer_and_autograd():
    # Hidden states: B=2, S=4, D=8
    np.random.seed(42)
    h_np = np.random.randn(2, 4, 8).astype(np.float32)
    h = from_numpy(h_np).requires_grad_(True)

    # Attention mask: doc 0 has 4 tokens, doc 1 has 2 tokens (last 2 are pad)
    mask = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    pooler = ContextualPoolingVectorizer(pooling="mean", normalize=True)
    pooled = pooler(h, attention_mask=mask)
    assert pooled.shape == (2, 8)

    # Verify L2 normalized
    p_norms = np.linalg.norm(pooled.to_numpy(), axis=1)
    np.testing.assert_allclose(p_norms, np.ones(2), rtol=1e-5, atol=1e-5)

    # Backward autograd check
    loss = (pooled * pooled).sum()
    loss.backward()
    assert h.grad is not None
    assert h.grad.shape == (2, 4, 8)
    assert not np.isnan(h.grad.to_numpy()).any()


def test_subword_tfidf_vectorizer():
    subword_vec = SubwordTfidfVectorizer(target_vocab_size=300, norm="l2")
    X = subword_vec.fit_transform(CORPUS)
    assert isinstance(X, Tensor)
    assert X.shape[0] == 4
    assert X.shape[1] >= 260

    # L2 norm check
    arr = X.to_numpy()
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5, atol=1e-5)

    # Test OOV and unseen word: should transform without crashing
    unseen = ["A quantum computer executes topological braids with anyons."]
    X_unseen = subword_vec.transform(unseen)
    assert X_unseen.shape == (1, X.shape[1])
    unseen_norm = np.linalg.norm(X_unseen.to_numpy()[0])
    assert abs(unseen_norm - 1.0) < 1e-4


def test_doc2vec_dbow_vectorizer():
    d2v = Doc2VecDBOWVectorizer(vector_size=32, epochs=8, lr=0.05, seed=42)
    doc_embs = d2v.fit_transform(CORPUS)
    assert isinstance(doc_embs, Tensor)
    assert doc_embs.shape == (4, 32)

    # Vectors should be unit normalized
    arr = doc_embs.to_numpy()
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-4, atol=1e-4)

    # Infer vector for a new query
    inferred = d2v.infer_vector("neural sequence models and attention")
    assert inferred.shape == (32,)
    inf_norm = np.linalg.norm(inferred)
    assert abs(inf_norm - 1.0) < 1e-4


def test_rust_core_fast_bm25_index():
    index = FastBM25Index.okapi()
    index.fit(CORPUS)
    assert index.num_docs == 4
    assert index.avgdl > 0.0

    scores = index.score_query("quick brown fox")
    assert len(scores) == 4
    assert scores[0] > 0.0  # Document 0 contains "quick brown fox"
    assert scores[1] == 0.0  # Document 1 does not

    top1 = index.top_k("neural sequence", 1)
    assert len(top1) == 1
    assert top1[0][0] == 3  # Document 3 is "Attention is all you need for neural sequence..."

    # BM25+
    index_plus = FastBM25Index.plus(1.0)
    index_plus.fit(CORPUS)
    scores_plus = index_plus.score_query("quick brown fox")
    assert scores_plus[0] > 0.0


def test_rust_core_fast_count_and_tfidf_vectorizers():
    # 1. FastCountVectorizer
    cv = FastCountVectorizer(ngram_range=(1, 2), binary=False, min_df=1)
    cv.fit(CORPUS)
    assert cv.vocab_size > 10

    csr = cv.transform(CORPUS)
    assert csr.shape == (4, cv.vocab_size)
    assert csr.nnz() > 0

    dense = cv.transform_dense(CORPUS)
    assert dense.shape == [4, cv.vocab_size]

    # 2. FastTfidfVectorizer
    tfidf = FastTfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, norm="l2")
    tfidf.fit(CORPUS)
    assert len(tfidf.idf) == tfidf.vocab_size

    tfidf_dense = tfidf.transform_dense(CORPUS)
    arr = tfidf_dense.to_numpy()
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-4, atol=1e-4)


def test_rust_core_fast_hashing_and_fasttext_char_ngrams():
    # 1. FastHashingVectorizer
    hv = FastHashingVectorizer(n_features=256, ngram_range=(1, 2), alternate_sign=True, norm="l2")
    csr = hv.transform(CORPUS)
    assert csr.shape == (4, 256)
    assert csr.nnz() > 0

    dense = hv.transform_dense(CORPUS)
    norms = np.linalg.norm(dense.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-4, atol=1e-4)

    # 2. FastTextCharNGramVectorizer
    ft = FastTextCharNGramVectorizer(dim=64, min_n=3, max_n=5)
    w1 = ft.transform_word("vectorization")
    w2 = ft.transform_word("vectorize")
    w3 = ft.transform_word("banana")

    dot12 = np.dot(w1, w2)
    dot13 = np.dot(w1, w3)
    assert dot12 > dot13  # Morphological similarity

    dense_docs = ft.transform_dense(CORPUS)
    assert dense_docs.shape == [4, 64]

    # 3. Native fast_murmurhash3_32
    h1 = fast_murmurhash3_32("oxideon", 42)
    assert h1 == 3622154769


def test_rust_core_kernel_spectral_vectorizer():
    rff = KernelSpectralVectorizer(in_dim=8, out_dim=32, gamma=0.5, seed=123)
    x_np = np.random.randn(3, 8).astype(np.float32)
    x = from_numpy(x_np)
    z = rff.transform(x)
    assert z.shape == [3, 32]
    assert not np.isnan(z.to_numpy()).any()


def test_rust_core_minhash_and_simhash_vectorizers():
    # 1. FastMinHashVectorizer
    mh = FastMinHashVectorizer(num_hashes=64, ngram_range=(1, 1), seed=42)
    assert mh.num_hashes == 64

    doc0 = CORPUS[0]
    doc0_similar = "The fast brown fox jumps over the lazy dog."
    doc1 = CORPUS[1]

    sig0 = mh.transform_document(doc0)
    sig0_sim = mh.transform_document(doc0_similar)
    sig1 = mh.transform_document(doc1)

    assert len(sig0) == 64
    jaccard_near = mh.jaccard_similarity(sig0, sig0_sim)
    jaccard_diff = mh.jaccard_similarity(sig0, sig1)
    assert jaccard_near > jaccard_diff
    assert jaccard_near > 0.5

    dense_mh = mh.transform_dense(CORPUS)
    assert dense_mh.shape == [4, 64]

    # 2. FastSimHashVectorizer
    sh = FastSimHashVectorizer(dim=64, ngram_range=(1, 1), seed=42)
    assert sh.dim == 64

    acc0, bits0 = sh.transform_document(doc0)
    acc0_sim, bits0_sim = sh.transform_document(doc0_similar)
    acc1, bits1 = sh.transform_document(doc1)

    dist_near = sh.hamming_distance(bits0, bits0_sim)
    dist_diff = sh.hamming_distance(bits0, bits1)
    assert dist_near < dist_diff

    dense_sh = sh.transform_dense(CORPUS)
    assert dense_sh.shape == [4, 64]
    norms = np.linalg.norm(dense_sh.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-4, atol=1e-4)


def test_dict_vectorizer_mixed_types():
    data = [
        {"city": "London", "temperature": 15.5, "rain": True},
        {"city": "Paris", "temperature": 22.0, "rain": False},
        {"city": "London", "temperature": 18.0, "humidity": 75},
    ]

    # Dense
    dv = DictVectorizer(sparse=False, sort=True)
    X = dv.fit_transform(data)
    assert isinstance(X, Tensor)
    assert X.shape[0] == 3
    assert X.shape[1] >= 5

    # Sparse
    dv_sparse = DictVectorizer(sparse=True, sort=True)
    X_sp = dv_sparse.fit_transform(data)
    assert isinstance(X_sp, CSRTensor)
    assert X_sp.shape == X.shape
    np.testing.assert_allclose(X_sp.to_dense().to_numpy(), X.to_numpy())

    # Inverse transform
    inv = dv.inverse_transform(X)
    assert len(inv) == 3
    assert inv[0]["city"] == "London"
    assert inv[0]["temperature"] == 15.5


def test_one_hot_vectorizer():
    colors = ["red", "green", "blue", "green", "red"]
    ohv = OneHotVectorizer(handle_unknown="ignore")
    X = ohv.fit_transform(colors)
    assert X.shape == [5, 3]

    # Verify rows sum to 1.0
    arr = X.to_numpy()
    np.testing.assert_allclose(np.sum(arr, axis=1), np.ones(5))

    # Test unknown
    X_unk = ohv.transform(["yellow", "blue"])
    assert X_unk.shape == [2, 3]
    assert np.sum(X_unk.to_numpy()[0]) == 0.0  # unknown ignored
    assert np.sum(X_unk.to_numpy()[1]) == 1.0  # known "blue"


def test_discretization_vectorizer():
    vals = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    disc = DiscretizationVectorizer(n_bins=4, strategy="uniform", one_hot=True)
    X = disc.fit_transform(vals)
    assert X.shape == [6, 4]
    np.testing.assert_allclose(np.sum(X.to_numpy(), axis=1), np.ones(6))

    # Ordinal
    disc_ord = DiscretizationVectorizer(n_bins=4, strategy="quantile", one_hot=False)
    X_ord = disc_ord.fit_transform(vals)
    assert X_ord.shape == [6, 1]


def test_json_tree_vectorizer():
    records = [
        {"user": {"name": "Alice", "age": 30, "address": {"city": "Berlin", "zip": 10115}}, "roles": ["admin", "dev"]},
        {"user": {"name": "Bob", "age": 25, "address": {"city": "Munich"}}, "roles": ["user"]},
    ]
    jtv = JsonTreeVectorizer(n_features=64, decay_rate=0.75, seed=42)
    X = jtv.transform(records)
    assert X.shape == [2, 64]
    norms = np.linalg.norm(X.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(2), rtol=1e-4, atol=1e-4)


def test_time_series_spectral_vectorizer():
    # 2 sample time series (sine wave and random noise)
    t = np.linspace(0, 1, 128)
    s1 = np.sin(2 * np.pi * 5 * t)
    s2 = np.random.RandomState(42).randn(128)

    ts_vec = TimeSeriesSpectralVectorizer(n_fft_bands=4, normalize=True)
    X = ts_vec.transform([s1, s2])
    assert X.shape[0] == 2
    assert X.shape[1] > 10
    norms = np.linalg.norm(X.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(2), rtol=1e-4, atol=1e-4)


def test_path_signature_vectorizer():
    # Trajectory 1: circular path in 2D
    theta = np.linspace(0, np.pi, 20)
    path1 = np.column_stack([np.cos(theta), np.sin(theta)])
    # Trajectory 2: linear path
    path2 = np.column_stack([theta, theta * 2.0])

    psv = PathSignatureVectorizer(depth=2)
    sigs = psv.transform([path1, path2])
    # depth 2 on 2D: level 1 is 2, level 2 is 4 -> total 6
    assert sigs.shape == [2, 6]
    assert not np.isnan(sigs.to_numpy()).any()


def test_spherical_harmonics_vectorizer():
    coords = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    sh_vec = SphericalHarmonicsVectorizer(degree=2)
    Y = sh_vec.transform(coords)
    assert Y.shape == [3, 9]  # (2+1)^2 = 9
    assert not np.isnan(Y.to_numpy()).any()


def test_deep_sets_vectorizer():
    # 2 sets of 3D items with different set sizes
    set1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # size 2
    set2 = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])  # size 3

    dsv = DeepSetsVectorizer(in_features=3, out_features=16, pooling="mean")
    X = dsv.transform([set1, set2])
    assert X.shape == [2, 16]

    # Permutation invariance: reordering set1 should yield identical vector
    set1_permuted = set1[::-1]
    X_perm = dsv.transform([set1_permuted])
    np.testing.assert_allclose(X.to_numpy()[0], X_perm.to_numpy()[0], atol=1e-5)


def test_weisfeiler_lehman_vectorizer():
    # Graph 1: Triangle (3 nodes fully connected)
    adj1 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    # Graph 2: Path (3 nodes in a line)
    adj2 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    wl = WeisfeilerLehmanVectorizer(iterations=2, n_features=32)
    X = wl.transform([(adj1, ["A", "A", "A"]), (adj2, ["A", "A", "A"])])
    assert X.shape == [2, 32]
    # Vectors should be non-identical because graphs are structurally distinct
    assert not np.allclose(X.to_numpy()[0], X.to_numpy()[1])


def test_colbert_maxsim_vectorizer():
    cmv = ColBERTMaxSimVectorizer(dim=16, seed=42)
    q_tokens = cmv.encode_tokens(["quick", "fox"])
    d_tokens = cmv.encode_tokens(["the", "quick", "brown", "fox", "jumps"])
    assert q_tokens.shape == [2, 16]
    assert d_tokens.shape == [5, 16]

    score = cmv.score_maxsim(q_tokens, d_tokens)
    assert score > 0.0

    # Batched
    q2_tokens = cmv.encode_tokens(["neural", "attention"])
    scores = cmv.batch_maxsim([q_tokens, q2_tokens], [d_tokens])
    assert scores.shape == [2, 1]


def test_byte_histogram_vectorizer():
    bhv = ByteHistogramVectorizer(log_scale=True, norm="l2")
    X = bhv.transform([b"binary payload 1", "text string 2"])
    assert X.shape == [2, 256]
    norms = np.linalg.norm(X.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, np.ones(2), rtol=1e-4, atol=1e-4)


def test_topic_model_vectorizer():
    # LSA
    lsa = TopicModelVectorizer(n_components=2, method="lsa")
    X_lsa = lsa.fit_transform(CORPUS)
    assert X_lsa.shape == [4, 2]
    norms_lsa = np.linalg.norm(X_lsa.to_numpy(), axis=1)
    # Docs 1..3 projected onto top 2 topics are unit normalized; orthogonal Doc 0 is 0
    np.testing.assert_allclose(norms_lsa[1:], np.ones(3), rtol=1e-4, atol=1e-4)
    assert abs(float(norms_lsa[0]) - 0.0) < 1e-4

    # NMF
    nmf = TopicModelVectorizer(n_components=2, method="nmf", max_iter=20)
    X_nmf = nmf.fit_transform(CORPUS)
    assert X_nmf.shape == [4, 2]
    assert (X_nmf.to_numpy() >= 0.0).all()



