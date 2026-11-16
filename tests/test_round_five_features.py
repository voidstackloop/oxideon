import pytest
import numpy as np
import oxideon as ox
from oxideon import (
    ssl,
    hpo,
    causal,
    privacy,
    genetic,
    graph,
    ensemble,
    rag,
    token_healing,
    anomaly,
)


def test_feature_1_ssl():
    """Tests PseudoLabelLoss, ConsistencyLoss, MeanTeacher, and SimCLRLoss."""
    # Pseudo-label loss
    pl = ssl.PseudoLabelLoss(threshold=0.8)
    logits_conf = ox.from_numpy(np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32))
    loss, ratio = pl(logits_conf)
    assert ratio == 1.0
    assert float(loss.to_numpy().reshape(-1)[0]) >= 0.0

    # Consistency loss
    cons = ssl.ConsistencyLoss(metric="mse")
    p1 = ox.randn([4, 4])
    p2 = ox.randn([4, 4])
    c_loss = cons(p1, p2)
    assert float(c_loss.to_numpy().reshape(-1)[0]) > 0.0

    # MeanTeacher
    student = ox.nn.Sequential(ox.nn.Linear(8, 4), ox.nn.Linear(4, 2))
    mt = ssl.MeanTeacher(student, alpha=0.9)
    mt.update(student)
    assert mt.teacher_params is not None
    mt.apply_to(student)

    # SimCLRLoss
    simclr = ssl.SimCLRLoss(temperature=0.1)
    z1 = ox.randn([4, 16])
    z2 = ox.randn([4, 16])
    sc_loss = simclr(z1, z2)
    assert float(sc_loss.to_numpy().reshape(-1)[0]) > 0.0


def test_feature_2_hpo():
    """Tests SearchSpace, Trial, RandomSearch, and GridSearch."""
    space = (
        hpo.SearchSpace()
        .choice("optimizer", ["adam", "sgd"])
        .uniform("lr", 1e-4, 1e-1)
        .int_uniform("batch_size", 16, 64)
    )
    sample = space.sample()
    assert sample["optimizer"] in ["adam", "sgd"]
    assert 1e-4 <= sample["lr"] <= 1e-1
    assert 16 <= sample["batch_size"] <= 64

    # Objective: maximize -(lr - 0.01)^2
    def obj(p):
        return -((p["lr"] - 0.01) ** 2)

    rs = hpo.RandomSearch(space, obj, n_trials=10, maximize=True, seed=42)
    rs.run()
    assert len(rs.trials) == 10
    assert rs.best_score is not None
    assert rs.best_params is not None

    # GridSearch
    grid = {"a": [1, 2], "b": [10, 20]}
    gs = hpo.GridSearch(grid, lambda p: p["a"] + p["b"], maximize=True)
    gs.run()
    assert len(gs.trials) == 4
    assert gs.best_score == 22.0
    assert gs.best_params == {"a": 2, "b": 20}


def test_feature_3_causal():
    """Tests average_treatment_effect, PropensityScoreMatcher, and DoubleMachineLearning."""
    # Synthetic observational data with true effect = 2.0
    n = 100
    t = np.random.binomial(1, 0.5, n)
    x = np.random.randn(n)
    y = 2.0 * t + 0.5 * x + np.random.randn(n) * 0.1

    # Difference in means ATE
    ate = causal.average_treatment_effect(y, t)
    assert 1.5 <= ate <= 2.5

    # Propensity matching
    matcher = causal.PropensityScoreMatcher(caliper=0.2)
    scores = 1.0 / (1.0 + np.exp(-x))  # pseudo propensity
    mt, mc = matcher.match(scores, t)
    assert len(mt) == len(mc)
    assert len(mt) > 0

    # Double Machine Learning
    dml = causal.DoubleMachineLearning()
    theta = dml.fit(y, t, x)
    assert 1.5 <= theta <= 2.5


def test_feature_4_privacy():
    """Tests clip_grad_per_sample, add_gaussian_noise, privacy_spent, and DP_SGD."""
    g = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
    clipped = privacy.clip_grad_per_sample(g, max_norm=1.0)
    assert np.isclose(np.linalg.norm(clipped), 1.0)

    noised = privacy.add_gaussian_noise(clipped, noise_multiplier=0.5, max_norm=1.0)
    assert noised.shape == (2,)

    # Privacy budget
    eps = privacy.privacy_spent(num_steps=1000, batch_size=32, total_dataset_size=10000, noise_multiplier=1.0)
    assert eps > 0.0

    # DP-SGD optimizer
    model = ox.nn.Sequential(ox.nn.Linear(4, 2))
    base_opt = ox.optim.Adam(model.parameters(), lr=0.01)
    dp_opt = privacy.DP_SGD(base_opt, noise_multiplier=0.1, max_grad_norm=1.0)

    x = ox.randn([2, 4])
    y = ox.from_numpy(np.array([0, 1], dtype=np.int64))
    dp_opt.zero_grad()
    loss = ox.cross_entropy(model(x), y)
    loss.backward()
    dp_opt.step()


def test_feature_5_genetic():
    """Tests GeneticAlgorithm and NeuroEvolution."""
    # GA minimizing sphere function f(x) = sum(x^2), so maximize -sum(x^2)
    ga = genetic.GeneticAlgorithm(population_size=20, genome_len=5, bounds=(-3.0, 3.0))
    best_ind = ga.evolve(lambda ind: -float(np.sum(ind**2)), generations=10)
    assert best_ind is not None
    assert len(best_ind) == 5

    # NeuroEvolution
    net = ox.nn.Sequential(ox.nn.Linear(4, 2))
    ne = genetic.NeuroEvolution(net, population_size=10, sigma=0.05, lr=0.02)
    score = ne.step(lambda m: -float(np.sum([np.sum(p.to_numpy() ** 2) for p in m.parameters()])))
    assert isinstance(score, float)


def test_feature_6_graph():
    """Tests laplacian_matrix, spectral_clustering, shortest_path_dijkstra, and pagerank."""
    # 4-node cycle graph 0 - 1 - 2 - 3 - 0
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=np.float32)

    # Laplacian
    l = graph.laplacian_matrix(adj, normalized=True)
    assert list(l.shape) == [4, 4]
    # Symmetrically normalized diagonal elements are 1.0 for connected nodes
    assert np.allclose(np.diag(l.to_numpy()), 1.0)

    # Spectral clustering
    labels = graph.spectral_clustering(adj, k=2)
    assert len(labels) == 4

    # Dijkstra shortest path from node 0
    dist = graph.shortest_path_dijkstra(adj, source=0)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    assert dist[2] == 2.0
    assert dist[3] == 1.0

    # PageRank
    ranks = graph.pagerank(adj).to_numpy()
    assert len(ranks) == 4
    # Symmetrical graph has uniform ranks
    assert np.allclose(ranks, 0.25, atol=1e-3)


def test_feature_7_ensemble():
    """Tests VotingClassifier, StochasticWeightAveraging, and model_soup."""
    m1 = ox.nn.Sequential(ox.nn.Linear(4, 2))
    m2 = ox.nn.Sequential(ox.nn.Linear(4, 2))
    ensemble_clf = ensemble.VotingClassifier([m1, m2], voting="soft")

    x = ox.randn([3, 4])
    probs = ensemble_clf.predict_proba(x)
    assert list(probs.shape) == [3, 2]
    preds = ensemble_clf.predict(x)
    assert len(preds) == 3

    # SWA
    swa = ensemble.StochasticWeightAveraging(m1)
    swa.update(m1)
    swa.update(m2)
    assert swa.num_averaged == 3

    # Model Soup
    soup = ensemble.model_soup([m1, m2], weights=[0.5, 0.5])
    assert soup is not None
    out_soup = soup(x)
    assert list(out_soup.shape) == [3, 2]


def test_feature_8_rag():
    """Tests Document, PromptTemplate, ContextCompressor, and RAGRetriever."""
    docs = [
        rag.Document("Oxideon supports automatic differentiation.", doc_id="doc1"),
        rag.Document("Oxideon is written in Rust and Python.", doc_id="doc2"),
    ]
    assert docs[0].doc_id == "doc1"
    assert "Document" in repr(docs[0])

    # Template
    tpl = rag.PromptTemplate("System: {system}\nContext: {context}\nUser: {query}")
    formatted = tpl.format(system="Be concise", context=docs[0].content, query="What is Oxideon?")
    assert "Be concise" in formatted
    assert "Oxideon supports automatic differentiation." in formatted

    # Compressor
    comp = rag.ContextCompressor(max_chars=100)
    compressed = comp.compress(docs)
    assert "doc1" in compressed

    # Retriever with VectorIndex
    idx = ox.search.VectorIndex(dim=4, metric="cosine")
    idx.add(ox.from_numpy(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)))
    retriever = rag.RAGRetriever(idx, docs)
    q = ox.from_numpy(np.array([[0.9, 0.1, 0, 0]], dtype=np.float32))
    results = retriever.retrieve(q, k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "doc1"


def test_feature_9_token_healing():
    """Tests repetition_penalty, top_k_top_p_filtering, greedy_search_step, and TokenBoundaryHealer."""
    logits = ox.from_numpy(np.array([[2.0, 5.0, 1.0]], dtype=np.float32))

    # Repetition penalty on token 1 (reduce 5.0 -> 5.0 / 1.5)
    penalized = token_healing.repetition_penalty(logits, past_tokens=[1], penalty=1.5)
    assert penalized.to_numpy()[0, 1] < 5.0

    # Top-K filtering (keep top 1)
    filtered = token_healing.top_k_top_p_filtering(logits, top_k=1)
    f_np = filtered.to_numpy()
    assert f_np[0, 1] == 5.0
    assert f_np[0, 0] <= -1e8

    # Greedy search step
    token = token_healing.greedy_search_step(logits, temperature=0.0)
    assert token == 1

    # Token boundary healer
    healer = token_healing.TokenBoundaryHealer()
    healed, suffix = healer.heal_prefix("The quick brown")
    assert healed == "The quick "
    assert suffix == "brown"


def test_feature_10_anomaly():
    """Tests EnergyScore, MahalanobisDistance, and IsolationForest."""
    # Energy score
    energy = anomaly.EnergyScore(temperature=1.0)
    logits_in = ox.from_numpy(np.array([[10.0, 2.0]], dtype=np.float32))
    logits_out = ox.from_numpy(np.array([[0.0, 0.0]], dtype=np.float32))

    e_in = energy.score(logits_in).to_numpy()
    e_out = energy.score(logits_out).to_numpy()
    # High confidence = negative large energy; low confidence = higher energy
    assert e_in[0] < e_out[0]

    # Mahalanobis distance
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = np.eye(2, dtype=np.float32)
    maha = anomaly.MahalanobisDistance(mean=mean, cov=cov)

    d_close = maha.score(ox.from_numpy(np.array([[0.1, 0.1]], dtype=np.float32))).to_numpy()
    d_far = maha.score(ox.from_numpy(np.array([[5.0, 5.0]], dtype=np.float32))).to_numpy()
    assert d_close[0] < d_far[0]

    # Isolation Forest
    X_train = np.random.randn(50, 2).astype(np.float32)
    iso = anomaly.IsolationForest(n_estimators=20, max_samples=32)
    iso.fit(X_train)

    normal_point = np.array([[0.0, 0.0]], dtype=np.float32)
    outlier_point = np.array([[10.0, 10.0]], dtype=np.float32)

    score_norm = iso.score(normal_point)[0]
    score_out = iso.score(outlier_point)[0]
    assert score_out > score_norm
