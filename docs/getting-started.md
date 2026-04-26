# Getting Started

This guide covers everything from cloning the repo to running a live recommendation API.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust (stable) | ≥ 1.75 | [rustup.rs](https://rustup.rs) |
| uv | any | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |

`uv` is only needed for downloading datasets. The Rust toolchain handles everything else.

---

## Step 1 — Clone and Build

```bash
git clone https://github.com/wedjaw/burn-recsys
cd burn-recsys

# Check everything compiles (fast — no training data needed)
cargo check

# Run unit tests (13 tests, ~1 second)
cargo test
```

Expected output:
```
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured
```

---

## Step 2 — Download Data

### Myket (Android app recommendations)

```bash
uv run python scripts/download_myket.py
# → data/myket.csv  (694,121 rows, ~35 MB)
```

### MovieLens 1M

```bash
uv run python scripts/download_movielens.py
# → data/movielens.csv  (1,000,209 rows, ~25 MB)
```

Validate the pipeline loaded correctly:

```bash
cargo run --example validate_data
```

Expected:
```
num_users  : 10000
num_items  : 7988
interactions: 694121
Data pipeline OK
```

---

## Step 3 — Inspect Model Architecture

```bash
cargo run --example model_info
```

```
=== Model Architecture ===

Dataset  : Myket (10000 users, 7988 items)

GMF
  user emb   : 10000 × 64
  item emb   : 7988 × 64
  output     : 64 → 1
  params     :    1151296

NeuMF (GMF path + MLP path)
  GMF user/item emb : 10000 × 64  |  7988 × 64
  MLP user/item emb : 10000 × 64  |  7988 × 64
  MLP layers        : 128→64→32→16
  output            : (64+16) → 1
  params            :    2313409
```

---

## Step 4 — Train

### NeuMF on Myket (default: 20 epochs)

```bash
cargo run --release --example myket_ncf
```

### Fewer epochs (for quick testing)

```bash
cargo run --release --example myket_ncf -- --epochs 5
```

### NeuMF on MovieLens

```bash
cargo run --release --example movielens_ncf -- --epochs 10
```

Training output (per epoch):
```
NeuMF params: 2313409
Epoch   1/5 | loss=0.362945 | 44.0s
Epoch   2/5 | loss=0.342207 | 44.4s
Epoch   3/5 | loss=0.338582 | 44.4s
Epoch   4/5 | loss=0.334605 | 43.6s
Epoch   5/5 | loss=0.330360 | 43.6s
Done. Checkpoints saved to checkpoints/
```

Checkpoints are saved as MessagePack files:
```
checkpoints/
├── epoch_001.mpk
├── epoch_002.mpk
...
└── epoch_005.mpk
```

### Training configuration

Edit `TrainConfig` defaults in `src/trainer/train.rs` or pass via the example:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Embedding vector size |
| `mlp_layers` | [128,64,32,16] | MLP hidden layer sizes |
| `learning_rate` | 0.001 | Adam learning rate |
| `batch_size` | 256 | Interactions per gradient step |
| `num_epochs` | 20 | Maximum training epochs |
| `neg_ratio` | 4 | Negative samples per positive |
| `patience` | 3 | Early stopping patience (epochs) |

---

## Step 5 — Evaluate

Compare GMF vs NeuMF with leave-one-out HR@10 and NDCG@10:

```bash
# Quick evaluation: 5 epochs, 500 users
cargo run --release --example evaluate -- --epochs 5 --users 500

# Full evaluation: 10 epochs, all test users
cargo run --release --example evaluate -- --epochs 10
```

Output:
```
╔══════════╦══════════╦══════════╗
║  Model   ║  HR@10   ║ NDCG@10  ║
╠══════════╬══════════╬══════════╣
║ GMF      ║ 0.1800   ║ 0.1028   ║
║ NeuMF    ║ 0.6040   ║ 0.4136   ║
╚══════════╩══════════╩══════════╝
```

---

## Step 6 — Serve

Start the recommendation API (requires a checkpoint from Step 4):

```bash
cargo run --release --bin server -- \
  --model checkpoints/epoch_005.mpk \
  --num-users 10000 \
  --num-items 7988 \
  --port 3000
```

```
INFO server: Model loaded (10000 users, 7988 items)
INFO server: Listening on http://0.0.0.0:3000
```

### API Reference

#### `GET /health`

```bash
curl http://localhost:3000/health
```

```json
{"status": "ok", "num_items": 7988}
```

#### `POST /recommend`

```bash
curl -X POST http://localhost:3000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 42, "candidates": [10, 20, 30, 40, 50]}'
```

```json
{
  "user_id": 42,
  "ranked": [50, 30, 10, 40, 20],
  "latency_ms": 0.492
}
```

**Request fields:**

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | u32 | Re-indexed user ID (0-based, from the dataset mapping) |
| `candidates` | Vec\<u32\> | Item IDs to score. If empty, ranks all items. |

**Note on IDs:** The server uses the re-indexed IDs produced by the data loader, not the original raw IDs from the CSV. To look up the mapping, use `MyketDataset::user_map` and `item_map` from the library.

---

## GPU Training (RTX 4060)

Switch backend by enabling the `cuda` feature:

```toml
# Cargo.toml
[features]
cuda = ["burn/tch"]
```

```rust
// In your example file, change:
type B = Autodiff<NdArray<f32>>;
// to:
type B = Autodiff<LibTorch<f32>>;
```

Build:
```bash
cargo run --release --features cuda --example myket_ncf
```

Requires LibTorch installed. Follow [Burn's CUDA setup guide](https://burn.dev/book/getting-started.html).

Expected speedup: ~5–10× per epoch on RTX 4060 vs M4 CPU.

---

## Adding a New Dataset

1. Implement `RecsysDataset`:

```rust
pub struct MyDataset { /* ... */ }

impl RecsysDataset for MyDataset {
    fn num_users(&self) -> usize { /* ... */ }
    fn num_items(&self) -> usize { /* ... */ }
    fn interactions(&self) -> &[(u32, u32)] { &self.interactions }
    fn train_test_split(&self, ratio: f32) -> (Self, Self) { /* ... */ }
}
```

2. Or use `PolarsDataset` directly if your data is a CSV:

```rust
let dataset = PolarsDataset::from_csv("data/my_data.csv", "user_id", "item_id")?;
```

3. The model, trainer, and evaluator require no changes.

---

## Common Issues

**`cargo: command not found`**
```bash
export PATH="$HOME/.cargo/bin:$PATH"
# or add to ~/.zshrc
```

**`Cannot open data/myket.csv`**

Run the download script first: `uv run python scripts/download_myket.py`

**`Failed to load checkpoint`**

Checkpoint files are generated during training. Run `cargo run --release --example myket_ncf -- --epochs 5` first.

**`user_id` mismatch between dataset and server**

The server uses 0-based re-indexed IDs, not raw CSV IDs. User `42` in the server refers to whichever raw user was assigned index 42 during loading. If you need the mapping, load the dataset and inspect `dataset.user_map`.

**OOM during training**

Reduce `batch_size` or `neg_ratio` in `TrainConfig`. With 555k train interactions and neg_ratio=4, each epoch processes 2.77M pairs. At batch_size=256, that is ~10,800 gradient steps.
