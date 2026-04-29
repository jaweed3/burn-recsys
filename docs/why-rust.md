# Why Rust?

The default choice for a recommendation system is Python — PyTorch, a Jupyter notebook, and a Flask API. That stack works fine for experimentation. This project was built in Rust instead, for reasons that become apparent once a system needs to be reliable, fast, and maintainable across its full lifecycle.

---

## Performance Without Compromise

Python ML systems typically operate in two modes: slow (training with pure Python orchestration) or fast (inference via compiled C++/CUDA kernels, called through ctypes bindings). The Python layer itself is overhead that cannot be eliminated.

Rust eliminates that layer entirely. The training loop, the data pipeline, and the HTTP server all run as native machine code with no interpreter, no GIL, and no garbage collector pauses.

**Measured latency for `/recommend` with 8 candidates:**
```
median: 0.49ms   p99: ~1.2ms   (NdArray backend, Apple M4 CPU)
```

A comparable Flask + PyTorch CPU serving stack would sit at 5–20ms for the same operation, primarily due to Python function call overhead and GIL contention under concurrent requests.

Rust's performance comes from:
- **Zero-cost abstractions** — generic code compiles to the same machine code as hand-written specializations
- **No GC pauses** — memory is freed deterministically at scope exit via RAII
- **LLVM backend** — the same optimizer used by Clang, with full auto-vectorization

---

## Memory Safety Without a Garbage Collector

Python and Java use tracing garbage collectors. GC pauses are invisible during development but cause latency spikes in production — exactly the wrong time for a recommendation API with a 100ms SLA.

Rust enforces memory safety at compile time through its ownership and borrow checker:
- Every value has exactly one owner
- References are either shared immutable (`&T`) or exclusive mutable (`&mut T`), never both
- No use-after-free, no double-free, no data races — guaranteed by the type system

The thread-safety constraints in this project (`NeuMF<NdArray>` not being `Sync` because `Param` uses `OnceCell`) were caught at compile time. The fix — one model clone per worker thread behind an mpsc channel — made the sharing contract explicit in the type. There is no equivalent mechanism in Python; the same bug manifests as a segfault or silent corruption in a multi-threaded NumPy workload.

---

## The Ecosystem Is Production-Ready

### Burn — training and inference in the same codebase

Burn is a deep learning framework designed from the ground up for Rust. Unlike PyTorch, which was built for Python and exposes C++ internals through bindings, Burn's entire stack is Rust: the autodiff engine, the tensor API, the backend abstraction.

The practical consequence: the same `NeuMF<B>` struct that is trained with `Autodiff<NdArray>` is loaded and run for inference with `NdArray` directly — no model export to ONNX, no separate TensorRT runtime, no format conversion. The type parameter `B: Backend` encodes which execution environment is active, and the compiler enforces that you don't accidentally run autodiff in the serving path.

Backend switching is a one-line change:
```rust
// Development (CPU, no dependencies)
type B = Autodiff<NdArray<f32>>;

// Production (CUDA via LibTorch)
type B = Autodiff<LibTorch<f32>>;
```

### Polars — predictable performance at scale

Polars is a DataFrame library built on Apache Arrow, written in Rust. Its lazy evaluation API compiles a query plan and executes it in a single multithreaded pass — no intermediate Python objects, no copies.

For 694k rows (Myket), Polars completes scan → cast → dedup → collect in under 200ms. The equivalent Pandas operation on the same data takes 2–4 seconds due to Python overhead and eager evaluation.

More importantly, Polars memory consumption is predictable. Pandas is notorious for making copies at unexpected moments (`.apply()`, type promotion, index alignment). In a production ETL pipeline where the dataset grows to 50M rows, unpredictable memory spikes cause OOM kills. Polars' columnar Arrow layout and explicit lazy evaluation mean memory usage scales linearly and predictably.

### Tokio — async without overhead

Tokio implements an async runtime using a work-stealing thread pool. Each Tokio task is a lightweight coroutine (~few KB of stack), not an OS thread (~1MB of stack). This means a single server process can handle thousands of concurrent `/recommend` requests without exhausting memory on thread stacks.

For recommendation serving, the bottleneck is not I/O (there is no disk or network access in the hot path) but CPU (model inference). The design uses an mpsc channel to distribute requests across a pool of worker threads, each holding its own model clone — no locks in the hot path. Tokio handles connection management and request parsing, then inference runs on dedicated workers in a two-stage pipeline: HNSW vector retrieval followed by neural scoring and ranking.

### OpenTelemetry — vendor-neutral observability

OpenTelemetry is a CNCF standard for traces, metrics, and logs. Using it from the start means the instrumentation works with any backend — stdout during development, Jaeger for distributed tracing, Prometheus + Grafana for dashboards — without changing application code.

The metrics collected here directly answer the questions that matter in production:
- `recsys.recommend.latency_ms` — is inference regressing as the model grows?
- `recsys.train.epoch_loss` — is training converging, or did a data change break it?
- `recsys.data.rows_loaded` — is the dataset pipeline ingesting the expected volume?

---

## What Rust Costs

Rust has real costs that should be acknowledged:

**Compile time.** A full release build of this project takes ~75 seconds on an M4. Incremental rebuilds after small changes take 1–3 seconds. Python has no compile step.

**Upfront type negotiation.** The embedding lookup bug (`Embedding::forward` expecting `Tensor<B, 2, Int>` but receiving `Tensor<B, 1, Int>`) fails at compile time with a clear error message. In PyTorch, the equivalent error is a runtime panic with a shape mismatch traceback. Rust's version requires a fix before you can even run the code; PyTorch's version lets you run for 30 minutes before failing at the first batch. This is the tradeoff: more friction upfront, zero surprises in production.

**Smaller ML ecosystem.** Burn has fewer pretrained models and less community tooling than PyTorch. For standard architectures like NeuMF and DeepFM that are implemented from scratch anyway, this is irrelevant. For tasks that require loading a pretrained transformer from HuggingFace, Python remains the practical choice.

---

## Summary

| Concern | Python + PyTorch | Rust + Burn |
|---|---|---|
| Iteration speed | Fast | Slower (compile step) |
| Inference latency | 5–20ms (CPU) | <1ms (CPU) |
| Memory predictability | Poor (GC, Pandas copies) | Excellent (RAII, Arrow) |
| Thread safety | Runtime errors / GIL | Compile-time enforcement |
| Deployment | Python runtime required | Single static binary |
| Ecosystem maturity | Very mature | Growing, sufficient for standard architectures |

The recommendation: use Python for exploration and prototyping. Use Rust when the system needs to be fast, correct, and maintainable in production.
