# Overview

`burn-recsys` is a production-grade recommendation system built entirely in Rust. It implements two neural collaborative filtering architectures — **NeuMF** and **DeepFM** — and ships with a full pipeline from raw data ingestion to a live HTTP serving API.

---

## What It Does

Given a user and a set of candidate items, the system scores each candidate and returns a ranked list. This is the core operation behind feed ranking, product recommendations, and app store personalization.

```
User #42 + Candidates [10, 20, 30, 40, 50] → [50, 30, 10, 40, 20]
                                                 ↑ ranked by predicted preference
```

The serving API handles this in **under 1 ms** per request on a single CPU core.

---

## Benchmark Results

Evaluated with leave-one-out protocol: for each user, one interaction is held out as ground truth, mixed with 99 random negatives. The model ranks all 100 candidates — we measure whether ground truth appears in the top 10.

| Model  | HR@10      | NDCG@10    | Params    |
|--------|-----------|-----------|-----------|
| GMF    | 0.180     | 0.103     | 1.15M     |
| NeuMF  | **0.604** | **0.414** | 2.31M     |

Dataset: [Myket Android App Recommendations](https://huggingface.co/datasets/erfanloghmani/myket-android-application-recommendation) — 10,000 users, 7,988 apps, 694,121 interactions. Training: 5 epochs, NdArray backend (Apple M4 CPU).

NeuMF exceeds the 0.6 HR@10 target despite training for only 5 epochs. Full 20-epoch training is expected to push NDCG@10 above 0.45.

---

## Datasets

| Dataset | Users  | Items  | Interactions | Download |
|---------|--------|--------|--------------|----------|
| Myket   | 10,000 | 7,988  | 694,121      | `uv run python scripts/download_myket.py` |
| MovieLens 1M | 6,040 | 3,706 | 1,000,209 | `uv run python scripts/download_movielens.py` |

Both datasets share the same `RecsysDataset` trait. Switching datasets requires changing one line in the example file — the model and training loop are untouched.

---

## Tech Stack

| Component      | Library                | Role                                   |
|----------------|------------------------|----------------------------------------|
| Model training | [Burn](https://burn.dev) 0.17 | Forward pass, backprop, checkpointing |
| Data pipeline  | [Polars](https://pola.rs) 0.46 | Fast lazy CSV loading, dedup, re-index |
| Serving API    | [Axum](https://github.com/tokio-rs/axum) 0.7 + [Tokio](https://tokio.rs) 1 | Async HTTP server |
| Observability  | [OpenTelemetry](https://opentelemetry.io) 0.22 | Metrics: latency, throughput, loss |

See [architecture.md](./architecture.md) for how these pieces connect, and [why-rust.md](./why-rust.md) for why Rust was chosen over Python.

---

## Project Structure

```
burn-recsys/
├── src/
│   ├── data/
│   │   ├── dataset.rs          # RecsysDataset trait
│   │   ├── myket.rs            # Myket CSV loader (csv crate)
│   │   ├── movielens.rs        # MovieLens 1M loader
│   │   ├── polars_loader.rs    # Fast Polars-backed loader
│   │   └── sampler.rs          # Negative sampler (training + eval)
│   ├── models/
│   │   ├── gmf.rs              # Generalized Matrix Factorization
│   │   ├── ncf.rs              # NeuMF (GMF + MLP paths)
│   │   └── deepfm.rs           # DeepFM (FM + Deep paths)
│   ├── metrics/
│   │   ├── eval.rs             # Leave-one-out evaluator
│   │   ├── hit_rate.rs         # HR@k
│   │   └── ndcg.rs             # NDCG@k
│   ├── trainer/
│   │   └── train.rs            # Adam + BCE + early stopping + checkpoints
│   ├── telemetry.rs            # OpenTelemetry setup
│   └── bin/
│       └── server.rs           # Axum HTTP server
├── examples/
│   ├── myket_ncf.rs            # Train NeuMF on Myket
│   ├── movielens_ncf.rs        # Train NeuMF on MovieLens
│   ├── evaluate.rs             # GMF vs NeuMF head-to-head
│   ├── model_info.rs           # Param counts
│   └── validate_data.rs        # Data pipeline smoke test
├── scripts/
│   ├── download_myket.py       # HuggingFace → CSV
│   └── download_movielens.py   # GroupLens → CSV
├── docs/
│   ├── overview.md             # This file
│   ├── architecture.md         # System design + math
│   ├── why-rust.md             # Language choice rationale
│   └── getting-started.md      # Build, train, serve
└── pyproject.toml              # uv project for Python scripts
```
