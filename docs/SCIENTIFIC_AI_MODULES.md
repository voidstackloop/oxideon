# Scientific AI, Structural Biology & Uncertainty Systems in Oxideon

Oxideon integrates domain-specialized deep learning architectures and statistical frameworks for causal discovery, molecular modeling, graph kernels, survival analysis, evidential uncertainty decomposition, and preference alignment.

---

## Table of Contents
1. [Causal Discovery & Structural Equation Modeling (`ox.causal_discovery`)](#1-causal-discovery--structural-equation-modeling)
2. [Structural Biology & Equivariant Networks (`ox.protein`)](#2-structural-biology--equivariant-networks)
3. [Graph Kernels & Optimal Transport Matching (`ox.graph_kernels` & `ox.graph_matching`)](#3-graph-kernels--optimal-transport-matching)
4. [Survival Analysis & Time-to-Event Modeling (`ox.survival`)](#4-survival-analysis--time-to-event-modeling)
5. [Evidential Deep Learning & Uncertainty Decomposition (`ox.uncertainty_quant`)](#5-evidential-deep-learning--uncertainty-decomposition)
6. [Post-Training Alignment & Preference Optimization (`ox.dpo`)](#6-post-training-alignment--preference-optimization)

---

## 1. Causal Discovery & Structural Equation Modeling

Continuous optimization of Directed Acyclic Graph (DAG) structures from observational data using the continuous acyclicity constraint (Zheng et al., NeurIPS 2018):

$$h(W) = \text{tr}\left(e^{W \odot W}\right) - d = 0$$

```python
import numpy as np
import oxideon as ox

# Generate synthetic linear SEM data: X = XW + noise
X = ox.Tensor.randn([1000, 5])

# Continuous NOTEARS DAG learning
learned_adj = ox.causal_discovery.notears_linear(
    X,
    lambda1=0.05,        # L1 sparsity penalty
    max_iter=50,
    w_threshold=0.2,     # Pruning threshold
)

# Verify graph acyclicity
is_dag = ox.causal_discovery.is_dag(learned_adj)
print("Is learned graph a valid DAG?", is_dag)

# Compute topological ordering of causal nodes
order = ox.causal_discovery.topological_sort(learned_adj)
print("Causal ordering:", order)
```

---

## 2. Structural Biology & Equivariant Networks

Provides representations for protein sequences, atomic contact topology, and $E(3)$/$SE(3)$-equivariant coordinate updates for geometric molecular modeling (Satorras et al., ICML 2021).

### Equivariant Graph Neural Network (EGNN) Layer
Guarantees strict rotational and translational equivariance for 3D atomic coordinates:

$$x_i' = x_i + \sum_{j \ne i} (x_i - x_j) \cdot \phi_x(m_{ij})$$

```python
import oxideon as ox

# Alpha-carbon backbone 3D coordinates (N atoms, 3)
ca_coords = ox.Tensor.randn([50, 3])
node_features = ox.Tensor.randn([50, 64])

# 1. High-throughput distance threshold contact map
contact_map = ox.compute_contact_map(ca_coords, threshold_angstrom=8.0)
print("Contact map shape:", contact_map.shape) # [50, 50]

# 2. E(3)-Equivariant Coordinate Update
egnn_layer = ox.EGNNCoordinateUpdate(node_dim=64, hidden_dim=32)
updated_h, updated_coords = egnn_layer(node_features, ca_coords)
print("Updated 3D coordinates:", updated_coords.shape) # [50, 3]
```

---

## 3. Graph Kernels & Optimal Transport Matching

### Weisfeiler-Lehman (WL) Subtree Kernel
1-WL color refinement produces multiset subtree histograms to compute topological similarity matrices between non-isomorphic graphs in $O(h(N + E))$:

```python
import oxideon as ox

# Adjacency matrices of two graphs
g1_adj = ox.erdos_renyi_graph(num_nodes=20, p=0.2)[0]
g2_adj = ox.erdos_renyi_graph(num_nodes=20, p=0.3)[0]

# Compute positive semi-definite Gram matrix K(G1, G2)
gram_matrix = ox.weisfeiler_lehman_kernel([g1_adj, g2_adj], num_iterations=3)
print("WL Gram Matrix (2x2):\n", gram_matrix.to_numpy())
```

### Random Walk Kernel (Kronecker-Decoupled)
Evaluates random walks on the direct product graph $A_1 \otimes A_2$ with geometric discount factor $\lambda$. Oxideon's decoupled formulation executes in $O(S(N_1^2 + N_2^2))$ instead of $O((N_1 N_2)^3)$:

$$\mathbf{1}^T (A_1 \otimes A_2)^k \mathbf{1} = (\mathbf{1}^T A_1^k \mathbf{1}) \cdot (\mathbf{1}^T A_2^k \mathbf{1})$$

```python
import oxideon as ox

g1_adj = ox.Tensor.eye(4)
g2_adj = ox.Tensor.eye(4)
similarity = ox.random_walk_kernel(g1_adj, g2_adj, num_steps=4, decay=0.05)
print("Random Walk Kernel Similarity:", similarity)
```

### Gromov-Wasserstein Optimal Transport
Solves non-convex quadratic assignment between graphs with different vertex counts using entropic Sinkhorn iterations:

```python
import oxideon as ox

g1_adj = ox.Tensor.eye(4)
g2_adj = ox.Tensor.eye(5)
transport_plan, gw_distance = ox.gromov_wasserstein_distance(
    cost1=g1_adj,
    cost2=g2_adj,
    num_iters=25,
    epsilon=0.05,
)
print("Gromov-Wasserstein Distance:", float(gw_distance.to_numpy().flat[0]))
```

---

## 4. Survival Analysis & Time-to-Event Modeling

Models censored time-to-event outcomes with partial likelihood objectives and ranking metrics.

```python
import oxideon as ox

risk_scores = ox.Tensor.randn([100, 1])
event_times = ox.Tensor.arange(1.0, 101.0, 1.0)
censoring_flags = ox.Tensor.ones([100]) # 1 = event occurred, 0 = censored

# 1. Cox Proportional Hazards Partial Log-Likelihood (Breslow approximation)
cox_loss_fn = ox.CoxProportionalHazardsLoss()
loss = cox_loss_fn(risk_scores, event_times, censoring_flags)

# 2. Harrell's Concordance Index (C-Index)
# Evaluates rank agreement between predicted risk and actual failure times
c_index = ox.concordance_index(risk_scores, event_times, censoring_flags)
print(f"Harrell's C-Index: {c_index:.3f}")

# 3. Non-Parametric Kaplan-Meier Estimator
km = ox.KaplanMeierEstimator()
unique_times, survival_probabilities = km.fit(event_times, censoring_flags)
```

---

## 5. Evidential Deep Learning & Uncertainty Decomposition

### Dirichlet Prior Loss
Places a Dirichlet prior over class probabilities to explicitly model prediction vacuity (lack of evidence) separately from aleatoric noise (Sensoy et al., NeurIPS 2018):

```python
import oxideon as ox

dirichlet_loss = ox.DirichletLoss(kl_weight=0.01)

logits = ox.Tensor.randn([16, 10])
targets_onehot = ox.Tensor.zeros([16, 10])

loss = dirichlet_loss(logits, targets_onehot)
```

### Information-Theoretic Uncertainty Decomposition
Given Monte Carlo dropout or ensemble probability outputs across $M$ stochastic forward passes, decomposes prediction variance into **Aleatoric** (data noise) and **Epistemic** (model ignorance):

$$\mathcal{I}(y; \theta \mid x) = \mathcal{H}\left(\mathbb{E}_{p(\theta)}[p(y|x, \theta)]\right) - \mathbb{E}_{p(\theta)}\left[\mathcal{H}(p(y|x, \theta))\right]$$

```python
import oxideon as ox

# [M ensemble models, B batch size, K classes]
ensemble_probs = ox.Tensor.rand([10, 32, 5])

total_entropy, aleatoric, epistemic = ox.decompose_uncertainty(ensemble_probs)
print("Epistemic Uncertainty Shape:", epistemic.shape) # [32]
```

---

## 6. Post-Training Alignment & Preference Optimization

Oxideon implements the primary reference-guided and reference-free objectives for aligning LLMs with human intent without requiring external reward models:

### Direct Preference Optimization (`ox.DPOLoss`)
Implicitly maximizes human preference margins between chosen $y_w$ and rejected $y_l$ responses (Rafailov et al., NeurIPS 2023):

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

```python
import oxideon as ox

pi_w = ox.Tensor.from_slice([-0.25, -0.15, -0.30], [3])
pi_l = ox.Tensor.from_slice([-1.80, -2.10, -1.45], [3])
ref_w = ox.Tensor.from_slice([-0.50, -0.80, -0.60], [3])
ref_l = ox.Tensor.from_slice([-1.20, -1.00, -1.60], [3])

dpo_loss = ox.DPOLoss(beta=0.1)

loss, chosen_rewards, rejected_rewards = dpo_loss(
    policy_chosen_logps=pi_w,
    policy_rejected_logps=pi_l,
    reference_chosen_logps=ref_w,
    reference_rejected_logps=ref_l,
)
```

### Odds Ratio Preference Optimization (`ox.ORPOLoss`)
Single-stage, reference-free preference alignment that combines standard supervised fine-tuning (SFT) with an odds-ratio penalty (Hong et al., 2024):

```python
import oxideon as ox

pi_w = ox.Tensor.from_slice([-0.25, -0.15, -0.30], [3])
pi_l = ox.Tensor.from_slice([-1.80, -2.10, -1.45], [3])

orpo_loss = ox.ORPOLoss(lambda_coeff=0.1)
loss_tensor, metrics = orpo_loss(pi_w, pi_l)
print("Odds ratio penalty:", metrics["odds_ratio_loss"])
```

### Kahneman-Tversky Optimization (`ox.KTOLoss`)
Aligns models directly from unpaired binary desirable/undesirable signals based on prospect theory with loss aversion weighting (Ethayarajh et al., 2024):

```python
import oxideon as ox

pi = ox.Tensor.from_slice([-0.20, -1.50, -0.35], [3])
ref = ox.Tensor.from_slice([-0.50, -0.80, -0.60], [3])

kto_loss = ox.KTOLoss(beta=0.1, desirable_weight=1.0, undesirable_weight=1.33)
labels = ox.Tensor.from_slice([1.0, 0.0, 1.0], [3])  # 1 = desirable, 0 = undesirable
loss, kto_metrics = kto_loss(policy_logps=pi, reference_logps=ref, labels=labels)
```
