# Architecture & Mathematics

This document covers the system architecture — how data flows from raw CSV to a ranked HTTP response — and the mathematical foundations of each model.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Data Layer (Polars)                         │
│                                                                      │
│  Raw CSV ──► LazyCsvReader ──► cast + dedup ──► collect             │
│                  │                                                   │
│                  └──► re-index (user_id, item_id → 0-based u32)     │
│                  └──► temporal sort by timestamp                    │
│                  └──► NegativeSampler (training + eval)             │
│                  └──► leave_one_out() temporal split                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Vec<(u32, u32)>
┌─────────────────────────────▼───────────────────────────────────────┐
│                         Model Layer (Burn)                           │
│                                                                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                           │
│  │   GMF   │   │  NeuMF  │   │ DeepFM  │  ← Scorable + Retrievable │
│  └─────────┘   └─────────┘   └─────────┘                           │
│                                                                      │
│  Training: Adam + BCELoss + val HR@k early stopping + .mpk ckpts   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ model loaded from best.mpk
┌─────────────────────────────▼───────────────────────────────────────┐
│                   Serving Layer (Axum + Tokio worker pool)           │
│                                                                      │
│  POST /recommend                                                     │
│   {"user_id": 42, "candidates": [...]}                              │
│         │                                                            │
│         ├── mpsc channel ──► worker pool (N × model clones)         │
│         ├── Stage 1: HNSW retrieval (user_vec → top-K items)       │
│         ├── Stage 2: model.score() ranking (precision scoring)      │
│         ├── sort by score descending                                 │
│         └── {"ranked": [...], "latency_ms": 0.49}                  │
│                                                                      │
│  GET /health    → {"status":"ok","model_type":"neumf","workers":8}  │
│  GET /ready     → {"ready":true,"workers":8}                        │
│  GET /swagger-ui → interactive API documentation                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ record_request() in hot path
┌─────────────────────────────▼───────────────────────────────────────┐
│                    Observability (OpenTelemetry)                      │
│                                                                      │
│  Counters:    recsys.recommend.requests (with model label)           │
│  Histograms:  recsys.recommend.latency_ms, recsys.train.epoch_loss  │
│  Export:      stdout → pipe to Prometheus / OTel Collector          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Re-indexing

Raw datasets have sparse, non-contiguous IDs (hashed integers, package names, etc.). Neural embedding tables require dense 0-based integer indices.

The loader builds two maps on the first pass through the file:

```
user_map: HashMap<original_id, u32>   // e.g. {-1185417981 → 0, -562407438 → 1, ...}
item_map: HashMap<String, u32>        // e.g. {"com.tencent.ig" → 0, ...}
```

This is O(n) in one scan — no sort, no second pass.

### Negative Sampling

The NCF paper uses 4 negative samples per positive interaction. During training, for each observed `(user, item+)` pair we sample 4 items the user has *not* interacted with:

```
sample(user_id) → Vec<item_id>  where  item_id ∉ user_positives[user_id]
```

For leave-one-out evaluation, we sample 99 negatives + 1 ground truth = 100 candidates per user.

---

## Model 1: Generalized Matrix Factorization (GMF)

GMF is the neural interpretation of classical matrix factorization.

**Embedding lookup:**

$$\mathbf{p}_u = \text{Emb}_\text{user}(u) \in \mathbb{R}^k$$
$$\mathbf{q}_i = \text{Emb}_\text{item}(i) \in \mathbb{R}^k$$

**Element-wise product (interaction):**

$$\phi_\text{GMF}(u, i) = \mathbf{p}_u \odot \mathbf{q}_i \in \mathbb{R}^k$$

**Output:**

$$\hat{y}_{ui} = \sigma\!\left(\mathbf{w}^\top (\mathbf{p}_u \odot \mathbf{q}_i)\right)$$

where $\sigma$ is sigmoid and $\mathbf{w} \in \mathbb{R}^k$ is a learned weight vector.

Classical MF is recovered when $\mathbf{w} = \mathbf{1}$ and $\sigma$ is the identity — so GMF strictly generalizes it.

---

## Model 2: NeuMF (Neural Matrix Factorization)

NeuMF fuses GMF with an MLP in two separate embedding spaces.

**Two separate embedding spaces:**

| Path | User embedding | Item embedding |
|------|---------------|---------------|
| GMF  | $\mathbf{p}_u^G \in \mathbb{R}^{k_G}$ | $\mathbf{q}_i^G \in \mathbb{R}^{k_G}$ |
| MLP  | $\mathbf{p}_u^M \in \mathbb{R}^{k_M}$ | $\mathbf{q}_i^M \in \mathbb{R}^{k_M}$ |

**GMF path** — captures linear pairwise interactions:

$$\phi_G = \mathbf{p}_u^G \odot \mathbf{q}_i^G$$

**MLP path** — captures non-linear, high-order interactions:

$$z_0 = \begin{bmatrix}\mathbf{p}_u^M \\ \mathbf{q}_i^M\end{bmatrix} \in \mathbb{R}^{2k_M}$$

$$z_l = \text{ReLU}(W_l z_{l-1} + b_l), \quad l = 1, \ldots, L$$

Layer sizes in this implementation: `[128, 64, 32, 16]` with $k_M = 64$, so $z_0 \in \mathbb{R}^{128}$.

**Fusion layer:**

$$\hat{y}_{ui} = \sigma\!\left(W_o \begin{bmatrix}\phi_G \\ z_L\end{bmatrix} + b_o\right)$$

with $W_o \in \mathbb{R}^{(k_G + h_L) \times 1}$, where $h_L = 16$ is the last MLP hidden size.

**Parameter count** (Myket: 10,000 users, 7,988 items, $k_G = k_M = 64$):

| Component | Parameters |
|-----------|-----------|
| GMF user/item emb | 10,000×64 + 7,988×64 = 1,150,848 |
| MLP user/item emb | 10,000×64 + 7,988×64 = 1,150,848 |
| MLP layers | 128×64 + 64×32 + 32×16 + 16 = 11,280 |
| Output layer | 80×1 + 1 = 81 |
| **Total** | **~2.31M** |

---

## Model 3: DeepFM

DeepFM replaces NeuMF's GMF path with a Factorization Machine (FM) component, which explicitly models pairwise feature interactions without manually engineering them.

**First-order terms** (bias per field):

$$y^{(1)} = b_u + b_i$$

where $b_u \in \mathbb{R}$ and $b_i \in \mathbb{R}$ are scalar bias embeddings.

**Second-order FM term** (pairwise interactions):

For $F$ fields with embedding vectors $\mathbf{v}_f \in \mathbb{R}^k$, the FM interaction is:

$$y^{(2)} = \frac{1}{2} \sum_{j=1}^{k} \left[\left(\sum_{f=1}^{F} v_{f,j}\right)^2 - \sum_{f=1}^{F} v_{f,j}^2\right]$$

For the two-field case (user + item) this simplifies to:

$$y^{(2)} = \mathbf{v}_u \cdot \mathbf{v}_i = \sum_{j=1}^k v_{u,j} \cdot v_{i,j}$$

This is computationally equivalent to the GMF element-wise product, summed across the embedding dimension — but it falls directly out of the FM formulation rather than being engineered by hand.

**Deep component** (same as NeuMF's MLP path, shared embeddings):

$$y^{\text{deep}} = W_{\text{out}} \cdot \text{MLP}\!\left(\begin{bmatrix}\mathbf{v}_u \\ \mathbf{v}_i\end{bmatrix}\right)$$

**Final output:**

$$\hat{y}_{ui} = \sigma\!\left(y^{(1)} + y^{(2)} + y^{\text{deep}}\right)$$

The key advantage over NeuMF: FM and Deep paths share the same embedding vectors $\mathbf{v}_u, \mathbf{v}_i$, so gradients from both paths update the same parameters. This leads to better utilization of the embedding space with fewer total parameters.

---

## Training Objective

Binary cross-entropy loss over positive and negative samples:

$$\mathcal{L} = -\frac{1}{|\mathcal{D}|} \sum_{(u,i,y) \in \mathcal{D}} \left[y \log \hat{y}_{ui} + (1 - y) \log(1 - \hat{y}_{ui})\right]$$

where $\mathcal{D}$ includes all positive interactions ($y = 1$) and sampled negatives ($y = 0$, ratio 4:1).

Optimizer: Adam with $\alpha = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.

---

## Evaluation Protocol

**Temporal leave-one-out** (standard in NCF literature):

1. Sort each user's interactions by timestamp ascending.
2. For each user, hide their temporally **last** interaction as ground truth.
3. Sample 99 random items the user has not interacted with.
4. Score all 100 candidates with the trained model.
5. Rank descending by score.
6. Measure HR@10 and NDCG@10.

**Hit Rate at k:**

$$\text{HR@k} = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}[\text{rank}(i_u^*) \leq k]$$

**Normalized Discounted Cumulative Gain at k:**

Since there is exactly one relevant item per user, IDCG = 1 and:

$$\text{NDCG@k} = \frac{1}{|U|} \sum_{u \in U} \frac{\mathbf{1}[\text{rank}(i_u^*) \leq k]}{\log_2(\text{rank}(i_u^*) + 1)}$$

A hit at rank 1 scores 1.0; at rank 2 scores $1/\log_2 3 \approx 0.631$; at rank 10 scores $1/\log_2 11 \approx 0.289$.

---

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). **Neural Collaborative Filtering**. *WWW'17*. [arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction**. *IJCAI'17*. [arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)
