# burn-recsys — Product Specification

> **Production-grade neural recommendation engine in Rust.**
> GMF + NeuMF + DeepFM. ~0.5ms inference. Two-stage retrieval + ranking. OpenTelemetry observability.

---

## 1. What It Is

**burn-recsys** is a full-stack deep learning recommendation system — from raw CSV ingestion to live HTTP serving — entirely in Rust. No Python, no Jupyter, no Flask. A single static binary that trains, evaluates, and serves neural collaborative filtering models with sub-millisecond latency.

It implements three state-of-the-art architectures from the NCF literature:

| Model | HR@10 | NDCG@10 | Params | Paper |
|-------|-------|---------|--------|-------|
| **GMF** | 0.180 | 0.103 | 1.15M | Neural interpretation of matrix factorization |
| **NeuMF** | **0.604** | **0.414** | 2.31M | He et al. WWW'17 |
| **DeepFM** | — | — | 1.19M | Guo et al. IJCAI'17 |

Benchmarked on Myket dataset (10K users, 7,988 items, 694K interactions). 5 epochs. CPU-only (Apple M4).

---

## 2. System Architecture

### End-to-End Pipeline

```
RAW CSV → Polars LazyFrame → Re-index + Dedup → Temporal Split
                                                      ↓
                                              Trainer (Adam + BCE)
                                                      ↓
                                              best.mpk Checkpoint
                                                      ↓
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                              HNSW ANN         Neural Ranker      Axum HTTP API
                           (instant-distance)  (sigmoid score)
                                    │                 │                 │
                                    └──── Two-stage retrieval + ranking ──┘
                                                                            │
                                                          POST /recommend ←─┘
                                                          GET /health
                                                          GET /ready
                                                          GET /swagger-ui
                                                                            │
                                                              OpenTelemetry ←┘
                                                          (latency + counters)
```

### Two-Stage Serving

```
POST /recommend {"user_id": 42}
         │
         ├── Stage 1: HNSW Retrieval (ANN)
         │     user_embedding → top-100 nearest item vectors
         │     Filters out already-interacted items
         │     ~0.12ms
         │
         ├── Stage 2: Neural Ranking
         │     model.score() → sigmoid → sort descending
         │     ~0.37ms
         │
         └── Response: {"ranked": [...], "latency_ms": 0.49}
```

### Worker Pool Architecture

- **mpsc channel** dispatches inference jobs to N worker threads
- Each worker holds its own model clone — **zero lock contention** in hot path
- Worker count = `available_parallelism()` (e.g., 8 on M4, 16+ on server)
- Optional client-provided candidates bypass retrieval stage for manual re-ranking

---

## 3. Core Components

### 3.1 Data Pipeline (`src/data/`)

| Component | File | Role |
|-----------|------|------|
| `RecsysDataset` trait | `dataset.rs` | Abstract interface — any dataset adapter |
| `PolarsDataset` | `polars_loader.rs` | CSV → lazy scan → cast → dedup → re-index → temporal sort |
| `NegativeSampler` | `sampler.rs` | Per-user negative sampling (training: 4:1, eval: 99 negatives) |

Key details:
- **Polars lazy evaluation** — single multithreaded pass, no copies. 694K rows in <200ms.
- **Re-indexing** — builds `user_index` + `item_index` hash maps in O(n), one scan. Raw sparse IDs → contiguous u32.
- **Temporal leave-one-out** — per-user timestamp sort guarantees the **last** interaction is held out as ground truth. Users with 1 interaction: appears in val only (embedding stays random).

### 3.2 Models (`src/models/`)

**GMF** (`gmf.rs`):
- Single embedding space: `p_u ⊙ q_i` (element-wise product)
- Linear output layer with sigmoid
- 1.15M params (user emb + item emb + output weight)

**NeuMF** (`ncf.rs`):
- Dual embedding spaces: GMF path + MLP path
- GMF path: separate embeddings, element-wise product
- MLP path: separate embeddings, concat → `[128, 64, 32, 16]` tower with ReLU
- Fusion: concat GMF output + MLP output → linear → sigmoid
- 2.31M params

**DeepFM** (`deepfm.rs`):
- Shared embedding vectors for FM + Deep paths
- FM first-order: `b_u + b_i` (scalar biases)
- FM second-order: `v_u · v_i` (dot product, two-field case)
- Deep path: concat → MLP tower → linear
- Combined: `sigmoid(FM + Deep)`
- 1.19M params (fewer than NeuMF — gradients from both paths update same embeddings)

**Model traits** (`mod.rs`):
- `Scorable<B>` — returns logits (pre-sigmoid). Generic over Burn backend.
- `Retrievable<B>` — exports `item_embeddings()` + `user_embedding()` for ANN indexing.
- `RecsysModel<B>` — composite trait: `Scorable + Retrievable + Send`.

### 3.3 Trainer (`src/trainer/`)

| Parameter | Default | Range |
|-----------|---------|-------|
| `embedding_dim` | 64 | 32–128 |
| `mlp_layers` | [128, 64, 32, 16] | configurable tower |
| `learning_rate` | 0.001 | Adam default |
| `batch_size` | 256 | constrained by memory |
| `epochs` | 20 | early-stop by patience |
| `neg_ratio` | 4 | negatives per positive |
| `patience` | 3 | epochs without HR@k improvement |

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss**: Binary cross-entropy on logits (numerically stable with clamp)
- **Validation**: Per-epoch HR@k + NDCG@k on 300 sampled users
- **Checkpointing**: Auto-saves `best.mpk` on HR@k improvement + `config.toml` for reproducibility
- **Experiment tracking**: `experiments.csv` (config hash, best metrics) + `experiments_epochs.csv` (per-epoch curves)

### 3.4 Serving Layer (`src/server/`)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | `{"status":"ok","model_type":"neumf","workers":8}` |
| `/ready` | GET | No | `{"ready":true,"workers":8}` — 503 until model loaded |
| `/recommend` | POST | `x-api-key` | Core recommendation endpoint |
| `/swagger-ui` | GET | No | Interactive OpenAPI docs (utoipa + Swagger UI) |

**Request/Response**:
```json
// POST /recommend
// Header: x-api-key: admin_bismillah
{"user_id": 42, "candidates": null}
// Response (200):
{"user_id": 42, "ranked": [160,49,73,1175,...], "latency_ms": 0.97}
```

**Input validation**:
- `user_id` out of range → 422
- Empty candidates array → 422
- Candidates > 200 items → 413
- Missing/invalid API key → 401
- Worker queue full → 500

### 3.5 Vector Retrieval (`src/server/retrieval.rs`)

- **Library**: `instant-distance` (HNSW ANN)
- **Index construction**: All item embeddings → HNSW graph at server startup
- **Search**: Euclidean distance (L2), approximate nearest neighbor
- **Exclusion**: Filter out items the user has already interacted with (from dataset positives)
- **Fallback**: Random sampling if no user vector available

### 3.6 Observability (`src/telemetry.rs`)

- **Framework**: OpenTelemetry SDK (vendor-neutral)
- **Export**: stdout (pipe to Prometheus/OTel Collector later)
- **Metrics**:
  - `recsys.recommend.requests` — counter by model label
  - `recsys.recommend.latency_ms` — histogram
  - `recsys.train.epoch_loss` — histogram
  - `recsys.data.rows_loaded` — counter
- **Tracing**: `tracing-subscriber` with env-filter, compact or JSON format

### 3.7 Middleware (`src/middleware/layer.rs`)

- API key authentication via `x-api-key` header
- Configured in `config/default.toml` (`valid_api_keys`)
- Applied to `/recommend` only (health/ready are public)

---

## 4. Performance

### Serving Latency (M4 CPU, 10 worker threads, k6 load test)

| Scenario | Avg | p50 | p90 | p95 | p99 |
|----------|-----|-----|-----|-----|-----|
| 3 VUs, random users | 1.29ms | 1.22ms | 1.88ms | 1.98ms | 2.13ms |
| Recommend with candidates | — | — | — | — | 0.12ms |
| Recommend (full ANN) | — | — | — | — | **0.97ms** |

**100% success rate.** Zero errors across all load test requests.

### Training Speed (M4 CPU)

- 5 epochs: ~4 minutes (Myket dataset, 694K interactions, neg_ratio=4)
- Per epoch: ~44 seconds (2.77M training pairs)
- GPU (RTX 4060 via LibTorch): expected 5-10× speedup

---

## 5. Backend Portability

The entire system is generic over Burn's backend parameter `B: Backend`:

```rust
// CPU development (default)
type B = Autodiff<NdArray<f32>>;

// CUDA production (requires --features cuda)
type B = Autodiff<LibTorch<f32>>;
```

The **same code** runs training and inference. No ONNX export. No TensorRT conversion. No format transfers. Change one type alias, recompile, done.

---

## 6. Data Support

| Dataset | Users | Items | Interactions | Adapter |
|---------|-------|-------|--------------|---------|
| Myket (Android Apps) | 10,000 | 7,988 | 694,121 | `PolarsDataset::myket()` |
| MovieLens 1M | 6,040 | 3,706 | 1,000,209 | `PolarsDataset::movielens()` |
| Any CSV | arbitrary | arbitrary | arbitrary | `PolarsDataset::from_csv()` |
| Custom source | arbitrary | arbitrary | arbitrary | `impl RecsysDataset` |

Adding a new dataset = implement 4 methods on a trait. Model, trainer, and server require zero changes.

---

## 7. Infrastructure

### Docker

Multi-stage Dockerfile:
1. **`cargo-chef` planner** — dependency caching layer
2. **Builder** — compile with `--release`, static binary output
3. **Runner** — `debian:12-slim`, non-root user, minimal image footprint

Entrypoint: `./server` (reads `config/default.toml`)

### CI/CD

GitHub Actions (`ci.yml`):
- `cargo check` + `cargo test` on every push
- 20 tests, ~2 seconds

### Configuration

TOML-based (`config/*.toml`) with env-var override (`APP_` prefix):
```toml
model = "checkpoints/best.mpk"
model_type = "neumf"
port = 3000
valid_api_keys = "admin_bismillah"
retrieval_limit = 100
max_candidates = 200
```

Separation of concerns: training config (`train_myket.toml`), eval config (`evaluate.toml`), server config (`default.toml`).

---

## 8. Project Structure

```
src/
├── data/           # Dataset loading, trait, negative sampling
│   ├── dataset.rs      # RecsysDataset trait
│   ├── polars_loader.rs # CSV → lazy → dedup → re-index → split
│   └── sampler.rs       # Negative sampling (train + eval)
├── models/         # Neural architectures
│   ├── mod.rs           # Scorable + Retrievable + RecsysModel traits
│   ├── gmf.rs           # Generalized Matrix Factorization
│   ├── ncf.rs           # NeuMF (GMF + MLP paths)
│   └── deepfm.rs        # DeepFM (FM + Deep paths)
├── metrics/        # Evaluation
│   ├── eval.rs          # Leave-one-out evaluator
│   ├── hit_rate.rs      # HR@k
│   └── ndcg.rs          # NDCG@k
├── trainer/        # Training loop
│   ├── config.rs        # TrainerSettings
│   └── train.rs         # Generic trainer (Adam + BCE + early stopping + logging)
├── server/         # HTTP serving layer
│   ├── mod.rs           # run() — worker pool init, channel dispatch
│   ├── handlers.rs      # /health, /ready, /recommend
│   ├── router.rs        # Axum router + Swagger UI
│   ├── state.rs         # AppState, Settings, InferenceJob
│   ├── model.rs         # Model loader (.mpk → RecsysModel)
│   └── retrieval.rs     # HNSW ANN retriever
├── middleware/      # Auth
│   ├── mod.rs
│   └── layer.rs         # API key middleware
├── telemetry.rs     # OpenTelemetry metrics + tracing
├── lib.rs           # Module exports
└── bin/server.rs    # Entrypoint

tests/
├── integration.rs   # Train → save → load → eval (full cycle)
└── server.rs        # HTTP: health, ready, recommend, validation

examples/
├── myket_ncf.rs     # Train NeuMF on Myket
├── movielens_ncf.rs # Train NeuMF on MovieLens
├── evaluate.rs      # GMF vs NeuMF head-to-head
├── model_info.rs    # Param counts
└── validate_data.rs # Data pipeline smoke test
```

---

## 9. Tech Stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Deep Learning | Burn | 0.17 | Forward pass, backprop, model definition |
| Backend (CPU) | Burn-NdArray | 0.17 | CPU tensor operations |
| Backend (GPU) | Burn-Tch (LibTorch) | — | CUDA via optional `cuda` feature |
| Data Pipeline | Polars | 0.46 | Lazy CSV loading, dedup, re-index |
| HTTP Server | Axum | 0.7 | Async routing, middleware |
| Async Runtime | Tokio | 1 | Work-stealing thread pool |
| ANN Index | instant-distance | 0.6 | HNSW vector retrieval |
| Observability | OpenTelemetry | 0.22 | Metrics pipeline |
| API Docs | Utoipa + Swagger UI | 4.1 / 6 | OpenAPI schema + interactive docs |
| Config | config | 0.14 | TOML + env-var config |
| Serialization | bincode | 1.3 | Checkpoint format |
| CLI | clap | 4 | (reserved for future use) |
| Load Testing | k6 | — | JavaScript-based load test script |
| Data Download | Python/uv | — | HuggingFace / GroupLens → CSV |

---

## 10. Why Rust > Python for This Stack

| Concern | Python + PyTorch | Rust + Burn |
|---------|-----------------|-------------|
| Inference latency (CPU) | 5–20ms | **<1ms** |
| Memory predictability | Poor (GC, Pandas copies) | **Excellent** (RAII, Arrow) |
| Thread safety | Runtime errors / GIL | **Compile-time enforcement** |
| Deployment | Python runtime + pip | **Single static binary** |
| Model export | ONNX → TensorRT pipeline | **Same code, recompile** |

The ~0.5ms median inference is not from kernel optimization — it's from eliminating the Python interpreter layer entirely. No GIL contention under concurrent requests. No GC pause latency spikes.

---

## 11. Key Design Decisions

1. **Type-parameterized model** (`NeuMF<B>`) — training uses `Autodiff<NdArray>`, inference uses `NdArray`. Compiler prevents autodiff in serving path.

2. **Worker pool via mpsc** — one model clone per thread, zero lock contention in hot path. Not `Arc<Mutex<Model>>`.

3. **Polars lazy evaluation** — single pass, no copies. Pandas would copy the DataFrame 3-4 times (cast, dedup, sort).

4. **Temporal leave-one-out** — not random split. Last interaction per user is the ground truth. This matters when user preferences drift over time.

5. **Separate train/eval/server configs** — TOML files committed to repo, env overrides for deployment. No config drift.

6. **OTel from day one** — vendor-neutral. `recsys.*` metric names prefixed for discoverability in any backend.

7. **HNSW + ranking as separate stages** — retrieval narrows 7,988 items → 100, ranking scores all 100. This two-stage pattern scales to millions of items.

8. **OpenAPI docs via utoipa** — schema annotations on DTOs generate `openapi.json` automatically. Swagger UI served at `/swagger-ui`.

---

## 12. Evaluation Protocol

**Leave-one-out** (standard per NCF literature):
1. Sort each user's interactions by timestamp ascending
2. Hide the **last** interaction per user as ground truth
3. Sample 99 random negatives (not in user's known positives)
4. Score 100 candidates, rank descending
5. Measure HR@k + NDCG@k

**Metrics**:
- **HR@k**: Did ground truth appear in top-k? Binary per user, averaged.
- **NDCG@k**: Where in top-k did it appear? Discounted by log(rank + 1).

Hit at rank 1 → NDCG = 1.0. Hit at rank 10 → NDCG = 0.289. This penalizes low-ranking hits even if they technically make HR@k.

---

## 13. Dataset Requirements

| Requirement | Format |
|-------------|--------|
| user column | Any hashable type (cast to string) |
| item column | Any hashable type (cast to string) |
| timestamp (recommended) | Numeric (f64 after cast) |
| duplicates | Auto-deduplicated (keeps first occurrence) |

Output: contiguous u32 indices for both users and items. All downstream components operate on these dense indices.

---

## 14. Constraints & Limits

- **Default item vocabulary**: 7,988 (Myket). Scales to ~10⁶ with HNSW (logarithmic search time).
- **Max candidates per request**: 200 (configurable via `max_candidates`)
- **Default retrieval limit**: 100 nearest neighbors (configurable via `retrieval_limit`)
- **Worker count**: `available_parallelism()` — typically 8-16 on modern hardware
- **Checkpoint format**: MessagePack `.mpk` via Burn's `CompactRecorder`

---

## 15. Reproducibility

- Training config saved as `checkpoints/config.toml` alongside `checkpoints/best.mpk`
- Experiment CSV logs config hash, parameters, and best metrics
- Deterministic random seed (`SmallRng::seed_from_u64(42)`) for reproducibility
- Temporal split guarantees consistent train/val boundaries given the same sorted data

---

## 16. Getting Started (7 commands)

```bash
git clone https://github.com/wedjaw/burn-recsys
cd burn-recsys
uv run python scripts/download_myket.py   # Get data
cargo run --release --example myket_ncf    # Train (4 min)
cargo run --release --bin server           # Serve
curl -X POST http://localhost:3000/recommend \
  -H 'x-api-key: admin_bismillah' \
  -d '{"user_id": 42}'                    # Recommend
```
