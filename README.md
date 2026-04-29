# burn-recsys

Production-grade recommendation system in Rust — GMF + NeuMF + DeepFM with Burn, Polars data pipeline, Axum serving API (two-stage retrieval + ranking), and OpenTelemetry observability.

[![CI](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml)

---

## Results

| Model  | HR@10  | NDCG@10 | Params |
|--------|--------|---------|--------|
| GMF    | 0.180  | 0.103   | 1.15M  |
| NeuMF  | **0.604** | **0.414** | 2.31M |

Myket dataset, 5 epochs, temporal leave-one-out eval (99 negatives + 1 ground truth per user).
Serving latency: **0.49ms** median per inference on CPU.

---

## Quick Start

```bash
# 1. Get data
uv run python scripts/download_myket.py

# 2. Train (5 epochs ~4 min on M4 CPU) — reads config/train_myket.toml
cargo run --release --example myket_ncf

# 3. Evaluate
cargo run --release --example evaluate

# 4. Serve — reads config/default.toml
cargo run --release --bin server

# 5. Request (with API key)
curl -X POST http://localhost:3000/recommend \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: admin_bismillah' \
  -d '{"user_id": 42, "candidates": [10, 20, 30, 40, 50]}'
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/overview.md](docs/overview.md) | What the project does, benchmark table, tech stack, project structure |
| [docs/architecture.md](docs/architecture.md) | System design, retrieval + ranking pipeline, math for GMF / NeuMF / DeepFM |
| [docs/why-rust.md](docs/why-rust.md) | Performance, memory safety, ecosystem — why Rust over Python |
| [docs/getting-started.md](docs/getting-started.md) | Step-by-step: build, download data, train, evaluate, serve, GPU setup |
| [docs/rust-ml-intern-guide.md](docs/rust-ml-intern-guide.md) | Complete Rust + ML intern guide — zero to reading this codebase |

---

## Stack

```
Polars  ──► data pipeline (lazy CSV, dedup, re-index, temporal split)
Burn    ──► model training + inference (NdArray CPU / LibTorch CUDA)
Axum    ──► HTTP serving (POST /recommend, GET /health, GET /ready)
OTel    ──► metrics: latency histogram, request counter (wired to hot path)
HNSW    ──► vector retrieval (instant-distance ANN index)
Swagger ──► interactive API docs at /swagger-ui
```

---

## References

- He et al. (2017). [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). *WWW'17*.
- Guo et al. (2017). [DeepFM](https://arxiv.org/abs/1703.04247). *IJCAI'17*.
