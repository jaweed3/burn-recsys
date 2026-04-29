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

# Run all tests (20 tests, ~2 seconds)
cargo test
```

Expected output:
```
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured
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
=== Model Architecture Info ===
Model Type: NeuMF
Parameters: 2313409
```

---

## Step 4 — Train

### NeuMF on Myket

```bash
cargo run --release --example myket_ncf
# Reads config/train_myket.toml. Override with env vars:
# APP_epochs=5 cargo run --release --example myket_ncf
```

### NeuMF on MovieLens

```bash
cargo run --release --example movielens_ncf
# Reads config/train_movielens.toml
```

Training output (per epoch):
```
NeuMF params: 2313409
Epoch   1/20 | loss=0.362945 | HR@10=0.4512 NDCG@10=0.2873 | 44.0s
Epoch   2/20 | loss=0.342207 | HR@10=0.5123 NDCG@10=0.3341 | 44.4s
...
Best epoch: 5 | HR@10=0.6041 | NDCG@10=0.4136
```

Checkpoints are saved as MessagePack files:
```
checkpoints/
├── config.toml     # training config for reproducibility
└── best.mpk         # best checkpoint by validation HR@10
```

### Training configuration

Edit the TOML config file (e.g., `config/train_myket.toml`) or override with env vars (`APP_` prefix):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Embedding vector size |
| `learning_rate` | 0.001 | Adam learning rate |
| `batch_size` | 256 | Interactions per gradient step |
| `epochs` | 20 | Maximum training epochs |
| `neg_ratio` | 4 | Negative samples per positive |
| `patience` | 3 | Early stopping on val HR@k (epochs) |
| `val_samples` | 300 | Users sampled per validation pass |
| `eval_k` | 10 | k for HR@k / NDCG@k tracking |

---

## Step 5 — Evaluate

Compare GMF vs NeuMF with temporal leave-one-out HR@10 and NDCG@10:

```bash
# Reads config/evaluate.toml
cargo run --release --example evaluate

# Override with env vars:
# APP_epochs=5 APP_users=500 cargo run --release --example evaluate
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

Start the recommendation API (reads `config/default.toml`, requires a trained checkpoint):

```bash
cargo run --release --bin server
```

```
INFO burn_recsys::server: Configuration: Settings { model: "checkpoints/best", model_type: "neumf", port: 3000, ... }
INFO burn_recsys::server: Model loaded and embeddings extracted (10000 users, 7988 items)
INFO burn_recsys::server: Inference worker 0 started
...
INFO burn_recsys::server: Listening on 0.0.0.0:3000
```

### API Reference

#### `GET /health`

```bash
curl http://localhost:3000/health
```

```json
{"status":"ok","num_users":10000,"num_items":7988,"model_type":"neumf","workers":8}
```

#### `GET /ready`

```bash
curl http://localhost:3000/ready
```

```json
{"ready":true,"workers":8}
```

#### `POST /recommend`

```bash
curl -X POST http://localhost:3000/recommend \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: admin_bismillah' \
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
| `candidates` | Option\<Vec\<u32\>\> | Item IDs to score. If `null`/omitted, uses two-stage retrieval (HNSW → ranking). |

**Authentication:** Requires `x-api-key` header (configured in `config/default.toml`).

**Swagger UI:** Interactive API docs at `http://localhost:3000/swagger-ui`.

**Note on IDs:** The server uses the re-indexed IDs produced by the data loader, not the original raw IDs from the CSV. Use `PolarsDataset::user_index` and `item_index` for the mapping.

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

1. Use `PolarsDataset::from_csv()` if your data is a CSV:

```rust
let dataset = PolarsDataset::from_csv(
    "data/my_data.csv",
    "user_id",          // user column name
    "item_id",          // item column name
    Some("timestamp"),  // optional timestamp for temporal split
)?;
let (train_ds, val) = dataset.leave_one_out();
```

2. Or implement `RecsysDataset` for a custom data source:

```rust
pub struct MyDataset { /* ... */ }

impl RecsysDataset for MyDataset {
    fn num_users(&self) -> usize { /* ... */ }
    fn num_items(&self) -> usize { /* ... */ }
    fn interactions(&self) -> &[(u32, u32)] { &self.interactions }
    fn train_test_split(&self, ratio: f32) -> (Self, Self) { /* ... */ }
}
```

3. The model, trainer, and evaluator require no changes — they work with any `RecsysDataset`.

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

Checkpoint files are generated during training. Run `cargo run --release --example myket_ncf` first. The best checkpoint is saved as `checkpoints/best.mpk`.

**`user_id` mismatch between dataset and server**

The server uses 0-based re-indexed IDs, not raw CSV IDs. User `42` in the server refers to whichever raw user was assigned index 42 during loading. If you need the mapping, load the dataset and inspect `dataset.user_index`.

**`401 Unauthorized` on /recommend**

Add the `x-api-key` header. Default key is `admin_bismillah` (set in `config/default.toml`).

**OOM during training**

Reduce `batch_size` or `neg_ratio` in `TrainConfig`. With 555k train interactions and neg_ratio=4, each epoch processes 2.77M pairs. At batch_size=256, that is ~10,800 gradient steps.
