# Rust for ML — A Complete Intern Guide

> This guide assumes you know at least one programming language (Python, JavaScript, anything).
> It will take you from zero Rust knowledge to reading and modifying this codebase confidently.
> Read it top to bottom once, then use it as a reference.

---

## Table of Contents

1. [What is Rust and why should you care?](#1-what-is-rust-and-why-should-you-care)
2. [The one idea that makes Rust different: Ownership](#2-the-one-idea-that-makes-rust-different-ownership)
3. [Syntax crash course](#3-syntax-crash-course)
4. [Traits — Rust's answer to interfaces and polymorphism](#4-traits--rusts-answer-to-interfaces-and-polymorphism)
5. [Generics — writing one function that works for many types](#5-generics--writing-one-function-that-works-for-many-types)
6. [Error handling — no exceptions, no null](#6-error-handling--no-exceptions-no-null)
7. [Iterators and closures](#7-iterators-and-closures)
8. [How ML works in Rust — the Burn framework](#8-how-ml-works-in-rust--the-burn-framework)
9. [Tensors in Burn](#9-tensors-in-burn)
10. [Modules — defining neural networks](#10-modules--defining-neural-networks)
11. [Training loop mechanics](#11-training-loop-mechanics)
12. [How this codebase is structured](#12-how-this-codebase-is-structured)
13. [Reading the code — annotated walkthrough](#13-reading-the-code--annotated-walkthrough)
14. [Common patterns you will see everywhere](#14-common-patterns-you-will-see-everywhere)
15. [Things that will confuse you (and why)](#15-things-that-will-confuse-you-and-why)
16. [Cheat sheet — syntax side by side with Python](#16-cheat-sheet--syntax-side-by-side-with-python)

---

## 1. What is Rust and why should you care?

### The short version

Rust is a systems programming language that runs as fast as C, but prevents the entire class of bugs that make C dangerous — use-after-free, data races, null pointer dereferences — **at compile time**, not at runtime.

### Why does this matter for ML?

Python ML is fast because the heavy lifting (matrix multiplication, CUDA kernels) is done in C/C++/CUDA underneath PyTorch or NumPy. Python is just the glue. The downside: Python is slow for everything else, has the GIL (prevents true multi-threading), and can't run on embedded systems or WASM.

Rust eliminates the Python middleman:

| What you want | Python way | Rust way |
|---|---|---|
| Fast tensor math | PyTorch (C++ backend) | Burn (Rust, same speed) |
| Data pipeline | Pandas (C++ core) | Polars (Rust core) |
| Serving API | FastAPI + Gunicorn | Axum + Tokio |
| Memory safety | GC / hope | Compiler guarantees |
| True parallelism | `multiprocessing` (ugly) | `rayon` (trivial) |

In this codebase, Rust handles everything end-to-end: data loading (Polars), model training (Burn), and HTTP serving (Axum). No Python needed.

### The cost

Rust has a steep learning curve. The compiler rejects code that other languages would happily run and then crash later. This is annoying for 2 weeks and then you realize it's actually checking your logic for you.

---

## 2. The one idea that makes Rust different: Ownership

Everything in Rust follows three rules. Once you internalize them, the compiler errors start making sense.

### Rule 1: Every value has exactly one owner

```rust
let s1 = String::from("hello"); // s1 owns the string
let s2 = s1;                    // ownership MOVES to s2
// println!("{s1}");             // ERROR: s1 no longer valid
println!("{s2}");               // OK
```

In Python, `s1` and `s2` would both point to the same object. In Rust, after the move, `s1` is gone. The compiler enforces this.

### Rule 2: You can borrow a value (read-only or read-write), but not both at once

```rust
let mut v = vec![1, 2, 3];

let r1 = &v;     // immutable borrow (read-only reference)
let r2 = &v;     // another immutable borrow — fine, multiple readers OK
println!("{r1:?} {r2:?}");

let r3 = &mut v; // mutable borrow (read-write reference)
r3.push(4);      // OK
// let r4 = &v;  // ERROR: can't borrow immutably while mutable borrow is active
```

The rule: **many readers OR one writer, never both simultaneously**. This is what prevents data races.

### Rule 3: A borrowed value cannot outlive the thing it borrows from

```rust
fn get_ref() -> &String {   // ERROR: returning a reference to what?
    let s = String::from("hello");
    &s                       // s will be dropped at the end of this function
}                            // the reference would dangle
```

The compiler tracks this and refuses to compile dangling references.

### Clone vs Move

If you want to keep using the original value, you have two options:

```rust
// Option 1: Clone (makes a deep copy, costs memory/time)
let s1 = String::from("hello");
let s2 = s1.clone();  // s1 still valid
println!("{s1} {s2}");

// Option 2: Use references (borrow without moving)
let s1 = String::from("hello");
let s2 = &s1;         // s2 borrows s1
println!("{s1} {s2}"); // both valid because s2 is just a view
```

### Copy types — the exception

Small, stack-allocated types (`i32`, `f32`, `bool`, `u32`) are `Copy` — they're duplicated automatically instead of moved:

```rust
let x = 42i32;
let y = x;        // x is copied, not moved
println!("{x}");  // still valid
```

---

## 3. Syntax crash course

### Variables

```rust
let x = 5;          // immutable (default)
let mut y = 5;      // mutable
y = 6;              // OK
// x = 6;           // ERROR: cannot assign to immutable variable

let z: f32 = 3.14;  // explicit type annotation
```

### Primitive types

```rust
i8, i16, i32, i64, i128, isize   // signed integers
u8, u16, u32, u64, u128, usize   // unsigned integers
f32, f64                          // floats
bool                              // true / false
char                              // unicode character
```

### Functions

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b  // no semicolon = return value (Rust is expression-based)
}

// Equivalent, explicit return:
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

### Structs

```rust
// Definition
struct User {
    name: String,
    age: u32,
    active: bool,
}

// Construction
let user = User {
    name: String::from("wedjaw"),
    age: 21,
    active: true,
};

// Access
println!("{}", user.name);

// Methods
impl User {
    // Associated function (no self) — like a static method / constructor
    fn new(name: &str, age: u32) -> Self {
        User {
            name: String::from(name),
            age,          // field shorthand when variable has same name
            active: true,
        }
    }

    // Method (takes &self — immutable reference to the instance)
    fn greet(&self) {
        println!("Hello, I'm {}", self.name);
    }

    // Mutable method (takes &mut self)
    fn deactivate(&mut self) {
        self.active = false;
    }
}

let mut u = User::new("wedjaw", 21);
u.greet();
u.deactivate();
```

### Enums

Rust enums are much more powerful than in other languages — each variant can hold different data.

```rust
// Simple enum
enum Direction { North, South, East, West }

// Enum with data (like a tagged union / sum type)
enum Shape {
    Circle(f64),           // radius
    Rectangle(f64, f64),  // width, height
    Triangle { base: f64, height: f64 }, // named fields
}

// The most important enum: Option<T> (replaces null)
enum Option<T> {
    Some(T),  // has a value
    None,     // no value
}

// The second most important enum: Result<T, E> (replaces exceptions)
enum Result<T, E> {
    Ok(T),   // success
    Err(E),  // failure
}
```

### Pattern matching

```rust
let shape = Shape::Circle(3.0);

match shape {
    Shape::Circle(r) => println!("Circle with radius {r}"),
    Shape::Rectangle(w, h) => println!("Rectangle {w}×{h}"),
    Shape::Triangle { base, height } => println!("Triangle base={base}"),
}

// Match must be exhaustive — all cases must be handled
// Use _ as a wildcard catch-all
match some_number {
    1 => println!("one"),
    2 | 3 => println!("two or three"),
    4..=9 => println!("four through nine"),
    _ => println!("something else"),
}
```

### if let / while let — shorter matching

```rust
let maybe_value: Option<i32> = Some(42);

// Instead of match:
if let Some(v) = maybe_value {
    println!("Got {v}");
}

// For loops
for i in 0..10 {         // 0 to 9
    println!("{i}");
}
for i in 0..=10 {        // 0 to 10 (inclusive)
    println!("{i}");
}

// Vectors
let mut v: Vec<i32> = Vec::new();
v.push(1);
v.push(2);
let v2 = vec![1, 2, 3];  // macro shorthand

for &x in &v {
    println!("{x}");
}
```

### Strings — two kinds

```rust
// &str — a string slice (borrowed, immutable view into string data)
let s: &str = "hello";         // lives in the binary, always valid

// String — an owned, heap-allocated, growable string
let s: String = String::from("hello");
let s: String = "hello".to_string();

// Converting
let owned: String = s.to_string(); // &str → String
let slice: &str = &owned;          // String → &str (borrow)

// Formatting
let name = "world";
let greeting = format!("Hello, {name}!"); // → String
println!("Hello, {name}!");               // → stdout
```

### Closures

Closures are anonymous functions that can capture variables from their surrounding scope.

```rust
let multiply_by = 3;
let times_three = |x| x * multiply_by;  // captures multiply_by
println!("{}", times_three(5));          // 15

// With explicit types
let add: fn(i32, i32) -> i32 = |a, b| a + b;

// Multi-line
let compute = |x: f32| {
    let y = x * 2.0;
    y + 1.0  // implicit return
};
```

### HashMap

```rust
use std::collections::HashMap;

let mut map: HashMap<String, u32> = HashMap::new();
map.insert("alice".to_string(), 42);
map.insert("bob".to_string(), 7);

// Access
if let Some(age) = map.get("alice") {
    println!("Alice is {age}");
}

// Insert if not present (common pattern)
let next_id = map.len() as u32;
let id = *map.entry("charlie".to_string()).or_insert(next_id);
```

---

## 4. Traits — Rust's answer to interfaces and polymorphism

A trait defines behavior. Any type can implement any trait. This is how Rust does polymorphism without classes or inheritance.

```rust
// Define a trait
trait Greet {
    fn hello(&self) -> String;

    // Default implementation (can be overridden)
    fn goodbye(&self) -> String {
        format!("Goodbye from {}", self.hello())
    }
}

// Implement for a type
struct Dog;
struct Cat;

impl Greet for Dog {
    fn hello(&self) -> String { "Woof!".to_string() }
}

impl Greet for Cat {
    fn hello(&self) -> String { "Meow!".to_string() }
    // goodbye() uses the default implementation
}

// Use it
let d = Dog;
println!("{}", d.hello());    // Woof!
println!("{}", d.goodbye());  // Goodbye from Woof!
```

### Derive macros — auto-implementing traits

Instead of writing `impl` by hand, common traits can be derived automatically:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub embedding_dim: usize,
    pub learning_rate: f64,
}
```

- `Debug` — allows `println!("{:?}", config)` for debugging
- `Clone` — adds a `.clone()` method that deep-copies the struct
- `Serialize`/`Deserialize` — allows conversion to/from JSON, TOML, CSV (from the `serde` crate)

### The Module trait in Burn

In this codebase, `#[derive(Module, Debug)]` on a neural network struct automatically implements the `Module` trait — which knows how to:
- Track all parameters (weights) for gradient computation
- Save/load checkpoints
- Switch between training and inference mode

---

## 5. Generics — writing one function that works for many types

```rust
// Without generics — only works for i32
fn largest_i32(list: &[i32]) -> i32 {
    let mut largest = list[0];
    for &item in list {
        if item > largest { largest = item; }
    }
    largest
}

// With generics — works for any type T that can be compared (PartialOrd)
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest { largest = item; }
    }
    largest
}
```

### Generic structs

```rust
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: std::fmt::Display> Pair<T> {
    fn show(&self) {
        println!("({}, {})", self.first, self.second);
    }
}
```

### Trait bounds — constraining generics

The `<T: SomeTrait>` syntax says "T must implement SomeTrait". Multiple bounds use `+`:

```rust
fn print_clone<T: Clone + std::fmt::Debug>(value: &T) {
    let copy = value.clone();
    println!("{copy:?}");
}

// Alternative syntax with where clause (cleaner for complex bounds)
fn train<D, M>(model: M, dataset: &D) -> M
where
    D: RecsysDataset,
    M: Scorable<B> + AutodiffModule<B> + Clone,
    M::InnerModule: Scorable<B::InnerBackend>,
{
    // ...
}
```

This is exactly the signature of the Trainer in this codebase. Read it as: "train works with any dataset D that implements RecsysDataset, and any model M that is Scorable, can be autodiff'd, and can be cloned."

### Associated types

Some traits have associated types — types that are part of the trait's contract:

```rust
trait AutodiffModule<B: AutodiffBackend>: Module<B> {
    type InnerModule: Module<B::InnerBackend>;
    fn valid(&self) -> Self::InnerModule;
}
```

When you write `M::InnerModule`, you're accessing the `InnerModule` associated type of trait `AutodiffModule` implemented for `M`. Think of it as a type alias that's defined per-implementation.

---

## 6. Error handling — no exceptions, no null

Rust has no `null` and no exceptions. Instead:
- Missing values → `Option<T>` (`Some(value)` or `None`)
- Fallible operations → `Result<T, E>` (`Ok(value)` or `Err(error)`)

### Option<T>

```rust
fn find_user(id: u32) -> Option<String> {
    if id == 0 {
        Some("Alice".to_string())
    } else {
        None
    }
}

// Pattern 1: match
match find_user(0) {
    Some(name) => println!("Found: {name}"),
    None => println!("Not found"),
}

// Pattern 2: if let
if let Some(name) = find_user(0) {
    println!("Found: {name}");
}

// Pattern 3: unwrap_or (provide a default)
let name = find_user(99).unwrap_or("Unknown".to_string());

// Pattern 4: ? (propagate None up, only in functions returning Option)
fn greet_user(id: u32) -> Option<String> {
    let name = find_user(id)?;  // returns None from greet_user if find_user returns None
    Some(format!("Hello, {name}!"))
}
```

### Result<T, E>

```rust
use std::fs;

fn read_config(path: &str) -> Result<String, std::io::Error> {
    fs::read_to_string(path)  // returns Result<String, io::Error>
}

// Pattern 1: match
match read_config("config.toml") {
    Ok(content) => println!("{content}"),
    Err(e) => println!("Error: {e}"),
}

// Pattern 2: ? operator (propagate errors up — used everywhere in this codebase)
fn load_and_parse(path: &str) -> anyhow::Result<()> {
    let content = read_config(path)?;  // if Err, return it immediately
    println!("{content}");
    Ok(())
}

// Pattern 3: unwrap (panics on Err — only for prototypes or tests)
let content = read_config("config.toml").unwrap();

// Pattern 4: expect (panic with message)
let content = read_config("config.toml").expect("could not read config");
```

### anyhow — the easy error handling library

This codebase uses `anyhow` for functions that can fail with any kind of error:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {                     // anyhow::Result = Result<T, anyhow::Error>
    let dataset = PolarsDataset::myket("data/myket.csv")
        .context("Failed to load Myket CSV")?; // adds context message on error
    Ok(())
}
```

---

## 7. Iterators and closures

Iterators in Rust are lazy — they don't do any work until consumed. This makes chains efficient.

```rust
let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// filter + map + collect
let result: Vec<i32> = numbers
    .iter()           // create iterator
    .filter(|&&x| x % 2 == 0)   // keep even numbers
    .map(|&x| x * x)             // square them
    .collect();       // consume into Vec
// result = [4, 16, 36, 64, 100]

// sum, count, max, min
let total: i32 = numbers.iter().sum();
let count = numbers.iter().filter(|&&x| x > 5).count();

// zip — pair up two iterators
let a = vec![1, 2, 3];
let b = vec!["one", "two", "three"];
let pairs: Vec<(i32, &str)> = a.iter().copied().zip(b.iter().copied()).collect();

// enumerate — get index + value
for (i, &x) in numbers.iter().enumerate() {
    println!("index={i}, value={x}");
}

// flat_map — map then flatten
let words = vec!["hello world", "foo bar"];
let all_words: Vec<&str> = words.iter()
    .flat_map(|s| s.split_whitespace())
    .collect();
// ["hello", "world", "foo", "bar"]
```

### Iterator reference patterns (important!)

```rust
let v = vec![1i32, 2, 3];

v.iter()           // yields &i32 (references)
v.iter().copied()  // yields i32 (copies — only for Copy types)
v.into_iter()      // yields i32 (consumes v — moves values out)

// That's why you see these patterns:
for &(uid, iid) in interactions { ... }   // destructure &(u32, u32) into u32, u32
for &x in &v { ... }                      // dereference
```

---

## 8. How ML works in Rust — the Burn framework

### What Burn is

Burn is a Rust ML framework similar to PyTorch. The key design choice: **the backend is a type parameter, not a runtime choice**.

```
PyTorch:  model = Model().to('cuda')   ← device chosen at runtime
Burn:     let model = Model::init::<Autodiff<Cuda>>(&device);  ← backend is in the type
```

This means Burn can catch backend-incompatible code at **compile time**, and zero-cost abstractions let the compiler optimize away all the generic machinery.

### Backends in this codebase

```rust
// Training backend — wraps NdArray with automatic differentiation
type B = Autodiff<NdArray<f32>>;

// Inference backend — plain NdArray, no autodiff overhead
type BInner = NdArray<f32>;
```

`NdArray` is a CPU backend backed by ndarray (Rust's NumPy equivalent). This codebase also supports CUDA via the `cuda` feature flag in Cargo.toml.

The `Autodiff<NdArray<f32>>` backend:
- Records all tensor operations in a computation graph
- Enables `.backward()` to compute gradients
- Has a method `model.valid()` that strips the autodiff wrapper and returns a plain `NdArray<f32>` model for inference

### The backends hierarchy

```
Autodiff<NdArray<f32>>    ← training (tracks gradients)
    └── NdArray<f32>      ← inference (fast, no gradient tracking)

AutodiffBackend trait     ← any backend with autodiff support
    └── Backend trait     ← any backend (for inference)
```

---

## 9. Tensors in Burn

### Creating tensors

```rust
use burn::tensor::{Tensor, Int, backend::Backend};

// Float tensor from slice
let t: Tensor<B, 1> = Tensor::from_floats(&[1.0, 2.0, 3.0], &device);
// Shape: [3]

// Integer tensor
let ids: Tensor<B, 1, Int> = Tensor::from_ints(&[0i32, 1, 2, 5], &device);

// Zeros and ones
let zeros: Tensor<B, 2> = Tensor::zeros([4, 64], &device);
let ones:  Tensor<B, 1> = Tensor::ones([100], &device);

// Random
let rand: Tensor<B, 2> = Tensor::random([3, 3], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
```

### Tensor type signature

`Tensor<B, D>` — B is the backend, D is the number of dimensions.
`Tensor<B, D, Int>` — integer tensor (for embedding indices).
`Tensor<B, D, Bool>` — boolean tensor (for masks).

### Common operations

```rust
let a: Tensor<B, 2> = Tensor::ones([3, 4], &device); // shape [3, 4]
let b: Tensor<B, 2> = Tensor::ones([3, 4], &device);

// Element-wise
let c = a.clone() + b.clone();
let d = a.clone() * b.clone();
let e = a.clone() - b.clone();

// Reduction
let sum:  Tensor<B, 1> = a.clone().sum_dim(1);    // sum along dim 1 → [3]
let mean: Tensor<B, 0> = a.clone().mean();         // scalar

// Reshape
let flat: Tensor<B, 1> = a.clone().reshape([12]);  // [3,4] → [12]
let view: Tensor<B, 3> = a.clone().reshape([3, 2, 2]); // [3,4] → [3,2,2]

// Unsqueeze — add a dimension
let a1: Tensor<B, 1> = Tensor::from_floats(&[1.0, 2.0], &device); // [2]
let a2: Tensor<B, 2> = a1.unsqueeze_dim(1); // [2, 1]

// Activation functions
use burn::tensor::activation::sigmoid;
let probs = sigmoid(logits); // element-wise sigmoid

// Clamp
let clamped = probs.clamp(1e-7, 1.0 - 1e-7);

// Extract scalar (only works on rank-0 tensors)
use burn::tensor::ElementConversion;
let loss_value: f32 = loss.into_scalar().elem::<f32>();

// Extract Vec<f32> from tensor
let scores_vec: Vec<f32> = scores.into_data().to_vec::<f32>().unwrap();
```

### Important: tensors are consumed on use (moved)

```rust
let a: Tensor<B, 1> = Tensor::ones([3], &device);
let b = a + a;  // ERROR: a is moved by the first use, can't use again

// Fix: clone before use
let a: Tensor<B, 1> = Tensor::ones([3], &device);
let b = a.clone() + a;  // OK — a.clone() is a separate tensor, a is moved after
```

This is the most common Burn-specific frustration. When in doubt, `.clone()` it.

---

## 10. Modules — defining neural networks

### The Module derive macro

```rust
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]  // Module derive auto-implements parameter tracking
pub struct GMF<B: Backend> {
    user_emb: Embedding<B>,
    item_emb: Embedding<B>,
    output:   Linear<B>,
}
```

The `#[derive(Module)]` macro inspects all fields and automatically:
- Implements parameter iteration (for gradient updates)
- Implements save/load (checkpoint serialization)
- Implements `valid()` (strips autodiff for inference)

### Configs — the constructor pattern

Burn uses a Config struct to initialize modules. This is how hyperparameters are stored separately from model weights:

```rust
#[derive(Debug, Clone)]
pub struct GMFConfig {
    pub num_users: usize,
    pub num_items: usize,
    pub embedding_dim: usize,
}

impl GMFConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GMF<B> {
        GMF {
            user_emb: EmbeddingConfig::new(self.num_users, self.embedding_dim).init(device),
            item_emb: EmbeddingConfig::new(self.num_items, self.embedding_dim).init(device),
            output:   LinearConfig::new(self.embedding_dim, 1).with_bias(false).init(device),
        }
    }
}
```

### Embedding — the key building block for recommendations

An `Embedding` layer is a lookup table: given an integer index, it returns a dense vector.

```rust
// EmbeddingConfig::new(vocab_size, embedding_dim)
let user_emb = EmbeddingConfig::new(10000, 64).init::<B>(&device);

// Forward pass: takes [batch] integer tensor, returns [batch, dim] float tensor
let user_ids: Tensor<B, 1, Int> = Tensor::from_ints(&[0i32, 1, 5], &device);
let user_ids_2d = user_ids.unsqueeze_dim(1); // [3] → [3, 1] (Embedding needs 2D)
let embeddings: Tensor<B, 3> = user_emb.forward(user_ids_2d); // [3, 1, 64]
let embeddings: Tensor<B, 2> = embeddings.squeeze(1);          // [3, 1, 64] → [3, 64]
```

### The forward pass

The `forward` method is the inference pass of the model:

```rust
impl<B: Backend> GMF<B> {
    pub fn forward(
        &self,
        user_ids: Tensor<B, 1, Int>,
        item_ids: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let n = user_ids.dims()[0];

        // Reshape for Embedding (needs 2D input)
        let u_emb = self.user_emb.forward(user_ids.unsqueeze_dim(1))
            .reshape([n, self.user_emb.num_embeddings()]);  // wait — use embedding_dim
        // Actually: squeeze the middle dim
        let u_emb = self.user_emb.forward(user_ids.reshape([n, 1]))
            .reshape([n, 64]);  // or use the configured dim

        let i_emb = self.item_emb.forward(item_ids.reshape([n, 1]))
            .reshape([n, 64]);

        // GMF: element-wise product
        let interaction = u_emb * i_emb;  // [n, dim]

        // Output: linear projection to scalar → [n]
        self.output.forward(interaction).reshape([n])
    }
}
```

### Training mode vs inference mode

```rust
// model is Autodiff<NdArray> during training (tracks computation graph)
let logits = model.score(user_t, item_t);
let loss = bce_loss(logits, label_t, n);
let grads = loss.backward();                           // compute gradients

// model.valid() strips autodiff → returns NdArray model for evaluation
let valid_model = model.valid();
let eval_logits = valid_model.score(eval_users, eval_items); // fast, no graph
```

---

## 11. Training loop mechanics

Here's the complete training loop skeleton, annotated:

```rust
// 1. Initialize
let mut optimizer = AdamConfig::new().init::<B, M>();   // Adam optimizer
let mut model = model_config.init::<B>(&device);         // random weights

// 2. Epoch loop
for epoch in 1..=num_epochs {
    // 3. Build training pairs (positive + negative samples)
    let mut pairs: Vec<(u32, u32, f32)> = Vec::new();
    for &(user, item) in interactions {
        pairs.push((user, item, 1.0));  // positive
        for neg in sampler.sample(user, &mut rng) {
            pairs.push((user, neg, 0.0)); // negative
        }
    }
    pairs.shuffle(&mut rng); // shuffle before mini-batching

    // 4. Mini-batch loop
    for chunk in pairs.chunks(batch_size) {
        // 4a. Build tensors from this batch
        let users:  Vec<i32> = chunk.iter().map(|&(u,_,_)| u as i32).collect();
        let items:  Vec<i32> = chunk.iter().map(|&(_,i,_)| i as i32).collect();
        let labels: Vec<f32> = chunk.iter().map(|&(_,_,l)| l).collect();

        let user_t  = Tensor::<B, 1, Int>::from_ints(&users, &device);
        let item_t  = Tensor::<B, 1, Int>::from_ints(&items, &device);
        let label_t = Tensor::<B, 1>::from_floats(&labels, &device);

        // 4b. Forward pass → compute loss
        let logits = model.score(user_t, item_t);
        let loss   = bce_loss(logits, label_t, batch_size);

        // 4c. Backward pass → get gradients
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // 4d. Update model weights
        model = optimizer.step(learning_rate, model, grads);
        // Note: model is returned, not mutated in-place (functional style)
    }
}
```

### Binary cross-entropy loss (BCE)

This is the standard loss for binary classification (interaction = 1 or 0):

```
BCE = -(y * log(p) + (1-y) * log(1-p))
```

Where `y` is the label (1 for real interaction, 0 for negative), `p` is the predicted probability (sigmoid of logit).

```rust
fn bce_loss<B: AutodiffBackend>(
    logits: Tensor<B, 1>,
    labels: Tensor<B, 1>,
    n: usize,
) -> Tensor<B, 1> {
    let eps = 1e-7_f32;
    let probs = sigmoid(logits).clamp(eps, 1.0 - eps);  // avoid log(0)
    let ones = Tensor::<B, 1>::ones([n], &probs.device());
    ((labels.clone() * probs.clone().log())
        + ((ones.clone() - labels) * (ones - probs).log()))
        .mean()
        .neg()  // negate because we minimize, not maximize
}
```

### Negative sampling

Real interactions are sparse. For a recommender, you have:
- Positives: (user, item) pairs the user actually interacted with
- Negatives: random (user, item) pairs the user probably hasn't seen

The model learns to score positives high and negatives low.

---

## 12. How this codebase is structured

```
burn-recsys/
├── src/
│   ├── lib.rs                  ← public API, re-exports everything
│   ├── data/
│   │   ├── mod.rs              ← module index
│   │   ├── dataset.rs          ← RecsysDataset trait (the interface)
│   │   ├── polars_loader.rs    ← PolarsDataset (Polars-backed CSV + temporal split)
│   │   └── sampler.rs          ← NegativeSampler
│   ├── models/
│   │   ├── mod.rs              ← Scorable + Retrievable traits + blanket impls
│   │   ├── gmf.rs              ← GMF model
│   │   ├── ncf.rs              ← NeuMF model
│   │   └── deepfm.rs           ← DeepFM model
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── eval.rs             ← evaluate() — temporal leave-one-out protocol
│   │   ├── hit_rate.rs         ← HR@k metric
│   │   └── ndcg.rs             ← NDCG@k metric
│   ├── trainer/
│   │   ├── config.rs           ← TrainerSettings (TOML-driven)
│   │   └── train.rs            ← Generic Trainer, TrainConfig, experiment log
│   ├── server/
│   │   ├── mod.rs              ← run() — worker pool, HNSW, two-stage pipeline
│   │   ├── handlers.rs         ← /health, /ready, /recommend + run_inference()
│   │   ├── router.rs           ← Axum router + Swagger UI
│   │   ├── state.rs            ← AppState, Settings, InferenceJob
│   │   ├── model.rs            ← load_model() for neumf/deepfm/gmf
│   │   └── retrieval.rs        ← VectorRetriever (HNSW) + CandidateGenerator trait
│   ├── middleware/
│   │   ├── mod.rs
│   │   └── layer.rs            ← API key auth middleware (x-api-key header)
│   ├── telemetry.rs            ← OpenTelemetry metrics + tracing
│   └── bin/
│       └── server.rs           ← Entrypoint: init OTel, load config, run server
├── examples/
│   ├── myket_ncf.rs            ← train NeuMF on Myket data
│   ├── movielens_ncf.rs        ← train NeuMF on MovieLens data
│   ├── evaluate.rs             ← compare GMF vs NeuMF
│   ├── validate_data.rs        ← sanity-check data pipeline
│   └── model_info.rs           ← print param counts for any model
├── tests/
│   ├── integration.rs          ← end-to-end train→save→load→eval test
│   └── server.rs               ← HTTP integration tests (health, recommend, errors)
├── config/                     ← TOML config files for each entrypoint
├── docs/                       ← all documentation
└── Cargo.toml                  ← dependencies and build config
```

### Module system

In Rust, each file is a module. You declare it in the parent:

```rust
// src/lib.rs
pub mod data;    // this file can now see src/data/mod.rs
pub mod models;
pub mod trainer;

// src/data/mod.rs
pub mod dataset;
pub mod polars_loader;
pub mod sampler;

pub use dataset::RecsysDataset;   // re-export: users can write `burn_recsys::data::RecsysDataset`
pub use polars_loader::PolarsDataset;
```

The `pub` keyword controls visibility. `pub` = anyone can use it. No `pub` = private to the module.

### Cargo.toml — the package manifest

```toml
[package]
name = "burn-recsys"
version = "0.2.0"
edition = "2021"

[dependencies]
burn = { version = "0.17", features = ["ndarray", "autodiff"] }
polars = { version = "0.46", features = ["csv", "lazy", "strings"] }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
anyhow = "1"
rand = { version = "0.8", features = ["small_rng"] }

[dev-dependencies]   # only for tests and examples
tempfile = "3"

[[example]]          # register each example file
name = "myket_ncf"
path = "examples/myket_ncf.rs"
```

---

## 13. Reading the code — annotated walkthrough

### The data pipeline (`src/data/polars_loader.rs`)

```rust
pub struct PolarsDataset {
    pub interactions: Vec<(u32, u32)>,  // all (user_idx, item_idx) pairs
    pub num_users: usize,
    pub num_items: usize,
    pub user_index: HashMap<String, u32>, // "user_123" → 0, "user_456" → 1, ...
    pub item_index: HashMap<String, u32>, // "com.foo" → 0, "com.bar" → 1, ...
    user_last_item: HashMap<u32, u32>,    // uid → last item (for eval holdout)
}
```

When you load a CSV, string user/item IDs get mapped to contiguous integers (0, 1, 2, ...) because neural network embeddings are just lookup tables indexed by integers.

```rust
// Loading
let dataset = PolarsDataset::myket("data/myket.csv")?;
// Sorted by timestamp, re-indexed to 0-based u32 IDs

// Temporal leave-one-out split
let (train_ds, val_interactions) = dataset.leave_one_out();
// train_ds: PolarsDataset with all interactions except temporally last per user
// val_interactions: Vec<(u32, u32)>, one per user — their most recent held-out item
```

### The model interface (`src/models/mod.rs`)

```rust
pub trait Scorable<B: Backend> {
    fn score(&self, users: Tensor<B, 1, Int>, items: Tensor<B, 1, Int>) -> Tensor<B, 1>;
}

pub trait Retrievable<B: Backend> {
    fn item_embeddings(&self) -> Vec<Vec<f32>>;
    fn user_embedding(&self, user_id: u32) -> Vec<f32>;
}

// Blanket impl: anything that's Scorable + Retrievable is a RecsysModel
pub trait RecsysModel<B: Backend>: Scorable<B> + Retrievable<B> + Send {}
```

Every model implements `Scorable` (for ranking) and `Retrievable` (for HNSW vector search). The Trainer only cares about `Scorable` — it doesn't know whether it's training a GMF, NeuMF, or DeepFM. This is **polymorphism via traits**.

### The trainer (`src/trainer/train.rs`)

```rust
pub fn train<D, M>(&self, model: M, train_ds: &D, val_interactions: &[(u32, u32)]) -> M
where
    D: RecsysDataset,
    M: Scorable<B> + AutodiffModule<B> + Clone,
    M::InnerModule: Scorable<B::InnerBackend>,
```

Read this where clause step by step:
- `D: RecsysDataset` — the dataset must provide `num_users()`, `num_items()`, `interactions()`
- `M: Scorable<B>` — the model can score (user, item) pairs during training
- `M: AutodiffModule<B>` — the model can compute gradients (has `.backward()` capability)
- `M: Clone` — we need to clone the best model state for early stopping
- `M::InnerModule: Scorable<B::InnerBackend>` — the inference-mode version (after `.valid()`) can also score

### Evaluation (`src/metrics/eval.rs`)

The leave-one-out evaluation protocol:

```
For each user:
  1. Take their held-out item (ground truth)
  2. Sample 99 random items they haven't seen (negatives)
  3. Score all 100 candidates with the model
  4. Rank descending by score
  5. Check if ground truth is in top K → HR@K
  6. Compute NDCG@K (penalizes ground truth ranking lower)
```

```rust
pub fn evaluate<B, D, F>(
    score_fn: F,         // closure: (users_tensor, items_tensor) → scores_tensor
    train_ds: &D,        // needed to know what NOT to sample as negatives
    test_interactions: &[(u32, u32)],  // held-out (user, item) pairs
    device: &B::Device,
    k: usize,
    max_users: usize,    // cap how many users to evaluate (speed/memory)
) -> EvalResult
```

### The server (`src/bin/server.rs` + `src/server/`)

```
POST /recommend  (requires x-api-key header)
  body: { "user_id": 42, "candidates": [1, 2, 3, 100] }
  returns: { "user_id": 42, "ranked": [100, 1, 3, 2], "latency_ms": 1.2 }

GET /health
  returns: { "status": "ok", "num_users": 10000, "num_items": 7988, "model_type": "neumf", "workers": 8 }

GET /ready
  returns: { "ready": true, "workers": 8 }

GET /swagger-ui  → interactive API docs
```

The server uses a two-stage pipeline: (1) HNSW vector retrieval to find top-K candidate items from user embedding similarity, then (2) neural scoring via the full model for precision ranking. Workers are spawned at startup — each holds its own model clone — and communicate via an `mpsc` channel, so there are no locks in the hot path.

---

## 14. Common patterns you will see everywhere

### The `?` operator — propagate errors upward

```rust
fn load_dataset(path: &str) -> anyhow::Result<PolarsDataset> {
    let ds = PolarsDataset::myket(path)?;  // if error, return it from this function
    Ok(ds)
}
```

Without `?`, every function call would need explicit match arms. `?` is just syntactic sugar for "if Err, return Err; if Ok, unwrap and continue."

### Shadowing — reuse variable names

```rust
let model = model_config.init::<B>(&device);   // GMF<Autodiff<NdArray>>
// ... train ...
let model = model.valid();                      // GMF<NdArray> — same name, different type
```

Rust allows re-declaring variables with the same name. The new binding shadows the old one. This is idiomatic for type transformations.

### Type aliases — shorten long types

```rust
type B = Autodiff<NdArray<f32>>;        // use B everywhere instead of the full type
type BInner = NdArray<f32>;

let model = model_config.init::<B>(&device);
```

### Turbofish syntax — explicit type parameters

```rust
// Normal: Rust infers the type
let v: Vec<i32> = vec![1, 2, 3].into_iter().collect();

// Turbofish: explicit type parameter on the method
let v = vec![1, 2, 3].into_iter().collect::<Vec<i32>>();

// On function calls:
let model = model_config.init::<B>(&device);  // B is the backend type
let model = model_config.init::<Autodiff<NdArray<f32>>>(&device);  // fully explicit
```

### `as` casting

```rust
let n: usize = 42;
let n_i32 = n as i32;
let n_f32 = n as f32;

// In this codebase:
let uid = user_id as i32;   // u32 → i32 for tensor creation
let idx = i as u32;         // usize → u32
```

### Slices — views into contiguous data

```rust
let v: Vec<i32> = vec![1, 2, 3, 4, 5];
let slice: &[i32] = &v;         // whole vec as slice
let partial: &[i32] = &v[1..3]; // [2, 3]

// Functions often take &[T] instead of &Vec<T> — more flexible
fn process(items: &[(u32, u32)]) { ... }
process(&v);              // works with Vec
process(&v[10..20]);      // works with partial slice
process(some_slice);      // works with any slice
```

### String formatting

```rust
println!("{}", value);           // Display (user-facing)
println!("{:?}", value);         // Debug (developer-facing, needs #[derive(Debug)])
println!("{value}");             // shorthand since Rust 2021
println!("{value:.4}");          // 4 decimal places
println!("{value:>8}");          // right-align in 8 chars
println!("{value:0>5}");         // zero-pad to 5 chars
println!("{epoch:>3}/{total}");  // right-align epoch
```

---

## 15. Things that will confuse you (and why)

### 1. "Value moved here" errors

```
error[E0382]: borrow of moved value: `tensor`
```

You used a tensor (or any non-Copy value) after it was moved. Fix: call `.clone()` before the first use that consumes it.

```rust
// Wrong
let loss = ((labels * probs.log()) + ((ones - labels) * (ones - probs).log())).mean();

// Right (ones and labels are used twice each)
let loss = ((labels.clone() * probs.clone().log())
    + ((ones.clone() - labels) * (ones - probs).log())).mean();
```

### 2. Cannot borrow as mutable because it is also borrowed as immutable

```
error[E0502]: cannot borrow `x` as mutable because it is also borrowed as immutable
```

You have an immutable reference and a mutable reference at the same time. Fix: make sure the immutable reference's lifetime ends before you take a mutable one.

### 3. The return type of Embedding forward

Burn's `Embedding::forward` takes `Tensor<B, 2, Int>` (not 1D!) and returns `Tensor<B, 3>`. You need to reshape:

```rust
// user_ids: Tensor<B, 1, Int>   shape [n]
let emb = self.user_emb.forward(user_ids.reshape([n, 1])); // input: [n, 1]
// emb: Tensor<B, 3>   shape [n, 1, dim]
let emb = emb.reshape([n, self.embedding_dim]);  // output: [n, dim]
```

### 4. `elem::<f32>()` — converting a scalar tensor to f32

```rust
use burn::tensor::ElementConversion;

let loss_f32: f32 = loss.into_scalar().elem::<f32>();
//                                     ^^^^ need ElementConversion in scope
```

### 5. `impl Trait` vs `dyn Trait`

```rust
// impl Trait — resolved at compile time (monomorphized, fast)
fn make_scorer() -> impl Fn(u32) -> f32 { ... }

// dyn Trait — resolved at runtime (dynamic dispatch, heap allocation, slower)
fn make_scorer() -> Box<dyn Fn(u32) -> f32> { ... }
```

In this codebase, we always use `impl Trait` (via generics) for the model to avoid runtime overhead.

### 6. Lifetime annotations

You'll rarely need to write lifetimes in this codebase, but you'll see them occasionally:

```rust
// 'a is a lifetime parameter — means "the output reference lives as long as the input reference"
fn first_word<'a>(s: &'a str) -> &'a str {
    s.split_whitespace().next().unwrap_or("")
}
```

The compiler infers most lifetimes automatically (lifetime elision). Only write them when the compiler asks you to.

### 7. `unwrap()` is fine in examples/tests, bad in production

```rust
// Fine in examples and tests (you control the inputs):
let dataset = PolarsDataset::myket("data/myket.csv").unwrap();

// Bad in library/server code (will crash on any error):
// Use ? instead
let dataset = PolarsDataset::myket("data/myket.csv")?;

// Or handle the error explicitly:
let dataset = match PolarsDataset::myket("data/myket.csv") {
    Ok(ds) => ds,
    Err(e) => {
        log::error!("Failed to load dataset: {e}");
        return Err(e.into());
    }
};
```

---

## 16. Cheat sheet — syntax side by side with Python

| Concept | Python | Rust |
|---|---|---|
| Variable | `x = 5` | `let x = 5;` |
| Mutable var | (all mutable) | `let mut x = 5;` |
| Type hint | `x: int = 5` | `let x: i32 = 5;` |
| Print | `print(f"val={x}")` | `println!("val={x}");` |
| Function | `def add(a, b): return a+b` | `fn add(a: i32, b: i32) -> i32 { a + b }` |
| Class | `class Foo:` | `struct Foo { ... }` + `impl Foo { ... }` |
| Constructor | `def __init__(self):` | `fn new(...) -> Self { ... }` |
| Method | `def greet(self):` | `fn greet(&self) { ... }` |
| Inheritance | `class Dog(Animal):` | Traits (composition, not inheritance) |
| Interface | `class ABC` / Protocol | `trait Trait { ... }` |
| Implement interface | `class Dog(Animal):` | `impl Animal for Dog { ... }` |
| Optional value | `x: Optional[int] = None` | `let x: Option<i32> = None;` |
| Check optional | `if x is not None:` | `if let Some(v) = x { ... }` |
| Exceptions | `try/except` | `Result<T, E>` + `?` operator |
| List | `[1, 2, 3]` | `vec![1, 2, 3]` |
| Dict | `{"a": 1}` | `HashMap::from([("a", 1)])` |
| List comprehension | `[x*2 for x in v if x>0]` | `v.iter().filter(\|&&x\| x>0).map(\|&x\| x*2).collect()` |
| f-string | `f"hello {name}"` | `format!("hello {name}")` |
| Type alias | `Tensor = torch.Tensor` | `type B = Autodiff<NdArray<f32>>;` |
| Decorator | `@dataclass` | `#[derive(Debug, Clone)]` |
| Import | `from foo import Bar` | `use foo::Bar;` |
| Module | file or `__init__.py` | `mod foo;` in parent + `foo.rs` file |
| Public | (default) | `pub` keyword |
| Private | `_name` convention | default (no `pub`) |
| Null safety | runtime error | compile error |
| Casting | `int(x)` | `x as i32` |
| String | `str` | `&str` (borrowed) or `String` (owned) |
| Copy semantics | by reference (shared) | explicit `.clone()` or `Copy` trait |
| Generics | `def f(x: List[T])` | `fn f<T>(x: &[T])` |
| For range | `for i in range(10):` | `for i in 0..10 {` |
| Enumerate | `for i, x in enumerate(v):` | `for (i, x) in v.iter().enumerate() {` |
| Zip | `zip(a, b)` | `a.iter().zip(b.iter())` |
| Match | `match x: case 1: ...` | `match x { 1 => ..., _ => ... }` |

### Cargo commands (your daily tools)

```bash
cargo check              # type-check without building — fastest, use often
cargo build              # build in debug mode (fast build, slow binary)
cargo build --release    # build in release mode (slow build, fast binary)
cargo test               # run all tests
cargo test -- --nocapture  # run tests and show println! output
cargo run --example myket_ncf  # run an example
cargo run --example myket_ncf -- --epochs 5  # run with args
cargo run --bin server   # run the server binary
cargo doc --open         # build and open documentation
cargo clippy             # linter — catches common mistakes
cargo fmt                # auto-format code
```

### Environment for this project

```bash
# Build and run a training example (reads config/train_myket.toml)
cargo run --release --example myket_ncf

# Override config with env vars
APP_epochs=5 cargo run --release --example myket_ncf

# Run all 20 tests
cargo test

# Start the serving API (reads config/default.toml)
RUST_LOG=info cargo run --release --bin server

# Enable JSON-formatted logs
LOG_FORMAT=json RUST_LOG=info cargo run --release --bin server
```

---

## Where to go from here

1. **The Rust Book** (free online at `doc.rust-lang.org/book`) — the definitive resource.  
   If you want depth on ownership, chapters 4-10 are essential.

2. **Rustlings** — small interactive exercises to practice syntax (`github.com/rust-lang/rustlings`).

3. **Burn docs** (`burn.dev/docs`) — reference for tensor ops, module API, backends.

4. **Polars docs** (`pola.rs`) — reference for the data pipeline API.

5. **Run the tests** and change things. The compiler will tell you exactly what's wrong and often suggest the fix. Rust's error messages are famous for being helpful.

> The most important thing: the Rust compiler is not your enemy. When it rejects your code, it's almost always right. Read the error message carefully — it usually tells you exactly what to do.
