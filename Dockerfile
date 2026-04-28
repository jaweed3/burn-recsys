# Stage 1: Install cargo-chef
FROM rust:1-slim as chef
WORKDIR /app
RUN cargo install cargo-chef

# Stage 2: Prepare the recipe
FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Build (cook) the dependencies
FROM chef as builder
COPY --from=planner /app/recipe.json recipe.json
# This stage will be cached unless Cargo.toml or Cargo.lock changes
RUN cargo chef cook --release --recipe-path recipe.json

# Stage 4: Build the actual application
# We use the same builder stage but now copy the source
COPY . .
RUN cargo build --release --bin server

# Stage 5: Create the final minimal image
FROM debian:12-slim as runner
WORKDIR /app

# Create a non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 appuser
USER appuser

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/server .

# Copy assets and configuration
USER root
COPY --chown=appuser:appuser checkpoints ./checkpoints
COPY --chown=appuser:appuser config ./config
USER appuser

# Expose port
EXPOSE 3000

ENTRYPOINT ["./server"]
