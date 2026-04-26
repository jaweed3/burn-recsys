# burn-recsys

Neural Collaborative Filtering (NCF / NeuMF) implemented from scratch with the [Burn](https://burn.dev) deep learning framework in Rust.

[![CI](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/wedjaw/burn-recsys/actions/workflows/ci.yml)

---

## Architecture

```
NeuMF (He et al., 2017)

User ID ─┬─► GMF User Emb (64) ─┐
          │                      ├─► element-wise ─► concat ─► Linear(80→1) ─► sigmoid
Item ID ─┼─► GMF Item Emb (64) ─┘       ↑
          │                              │
          ├─► MLP User Emb (64) ─┐       │
          │                      ├─► concat → 128→64→32→16 (ReLU) ─┘
          └─► MLP Item Emb (64) ─┘
```

Two separate embedding spaces: GMF path captures linear interactions, MLP path captures non-linear patterns. Final layer concatenates both.

---

## Benchmark Results (Myket, 5 epochs, 500 users eval)

| Model  | HR@10  | NDCG@10 |
|--------|--------|---------|
| GMF    | 0.1800 | 0.1028  |
| NeuMF  | 0.6040 | 0.4136  |

Eval protocol: leave-one-out with 99 random negatives + 1 ground truth per user.

Dataset: [`erfanloghmani/myket-android-application-recommendation`](https://huggingface.co/datasets/erfanloghmani/myket-android-application-recommendation) — 10k users, 7.9k apps, 694k interactions.

---

## Quick Start

### 1. Prerequisites

- Rust (stable) — [rustup.rs](https://rustup.rs)
- [uv](https://docs.astral.sh/uv/) — Python package manager (for data download only)

### 2. Download data

```bash
# Myket (HuggingFace, requires Python/uv)
uv run python scripts/download_myket.py

# MovieLens 1M (no Python needed beyond stdlib)
uv run python scripts/download_movielens.py
```

### 3. Train

```bash
# Myket — NeuMF, 20 epochs
cargo run --release --example myket_ncf

# MovieLens — same model, same command
cargo run --release --example movielens_ncf

# Fewer epochs for quick test
cargo run --release --example myket_ncf -- --epochs 5
```

### 4. Evaluate GMF vs NeuMF

```bash
cargo run --release --example evaluate -- --epochs 5 --users 500
```

---

## Project Structure

```
burn-recsys/
├── src/
│   ├── data/
│   │   ├── dataset.rs      ← RecsysDataset trait
│   │   ├── myket.rs        ← Myket CSV loader
│   │   ├── movielens.rs    ← MovieLens 1M loader
│   │   └── sampler.rs      ← NegativeSampler
│   ├── models/
│   │   ├── gmf.rs          ← GMF baseline
│   │   └── ncf.rs          ← NeuMF (GMF + MLP)
│   ├── metrics/
│   │   ├── eval.rs         ← leave-one-out evaluator
│   │   ├── hit_rate.rs     ← HR@k
│   │   └── ndcg.rs         ← NDCG@k
│   └── trainer/
│       └── train.rs        ← Adam + BCE + early stopping + checkpoints
├── examples/
│   ├── myket_ncf.rs        ← Myket training entry point
│   ├── movielens_ncf.rs    ← MovieLens training entry point
│   ├── evaluate.rs         ← GMF vs NeuMF comparison
│   ├── model_info.rs       ← param counts
│   └── validate_data.rs    ← data pipeline smoke test
├── scripts/
│   ├── download_myket.py
│   └── download_movielens.py
├── pyproject.toml          ← uv project for Python scripts
└── .github/workflows/ci.yml
```

---

## Adding a New Dataset

Implement `RecsysDataset` — that's it. The model and trainer are dataset-agnostic.

```rust
pub trait RecsysDataset {
    fn num_users(&self) -> usize;
    fn num_items(&self) -> usize;
    fn interactions(&self) -> &[(u32, u32)];
    fn train_test_split(&self, ratio: f32) -> (Self, Self) where Self: Sized;
}
```

---

## Backend Switching (CPU → GPU)

Default backend: `NdArray` (CPU, no dependencies).

For CUDA training on RTX 4060:

```toml
# Cargo.toml
burn = { version = "0.17", features = ["tch"] }
```

```rust
// In your example
type B = Autodiff<LibTorch<f32>>;
```

---

## Reference

He, X. et al. (2017). *Neural Collaborative Filtering*. WWW'17.
[arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031)
