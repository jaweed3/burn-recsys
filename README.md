# burn-recsys

Production-grade recommendation system in Rust — NeuMF + DeepFM with Burn, Polars data pipeline, Axum serving API, and OpenTelemetry observability.

[![CI](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml)

---

## Results

| Model  | HR@10  | NDCG@10 | Params |
|--------|--------|---------|--------|
| GMF    | 0.180  | 0.103   | 1.15M  |
| NeuMF  | **0.604** | **0.414** | 2.31M |

Myket dataset, 5 epochs, leave-one-out eval (99 negatives + 1 ground truth per user).
Serving latency: **0.49ms** median on a single CPU core.

---

## Quick Start

```bash
# 1. Get data
uv run python scripts/download_myket.py

# 2. Train (5 epochs ~4 min on M4 CPU)
cargo run --release --example myket_ncf -- --epochs 5

# 3. Evaluate
cargo run --release --example evaluate -- --epochs 5 --users 500

# 4. Serve
cargo run --release --bin server -- --model checkpoints/epoch_005.mpk

# 5. Request
curl -X POST http://localhost:3000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 42, "candidates": [10, 20, 30, 40, 50]}'
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/overview.md](docs/overview.md) | What the project does, benchmark table, tech stack, project structure |
| [docs/architecture.md](docs/architecture.md) | System design diagram, math for GMF / NeuMF / DeepFM, loss function, eval protocol |
| [docs/why-rust.md](docs/why-rust.md) | Performance, memory safety, ecosystem — why Rust over Python for this use case |
| [docs/getting-started.md](docs/getting-started.md) | Step-by-step: build, download data, train, evaluate, serve, GPU setup |

---

## Stack

```
Polars  ──► data pipeline (lazy CSV, dedup, re-index)
Burn    ──► model training + inference (NdArray CPU / LibTorch CUDA)
Axum    ──► HTTP serving API (POST /recommend, GET /health)
OTel    ──► metrics: latency histogram, request counter, epoch loss
```

---

## References

- He et al. (2017). [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). *WWW'17*.
- Guo et al. (2017). [DeepFM](https://arxiv.org/abs/1703.04247). *IJCAI'17*.
