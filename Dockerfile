# Stage 1: Build the application
# We use cargo-chef to cache dependencies and speed up builds
FROM rust:1-slim as chef
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
# Create a recipe for the dependencies
RUN cargo chef prepare --recipe-path recipe.json

FROM chef as planner
COPY --from=chef /app/recipe.json recipe.json
# Build the dependencies
RUN cargo chef cook --release --recipe-path recipe.json

# --- Builder Stage ---
FROM rust:1-slim as builder
WORKDIR /app
# Copy the pre-built dependencies
COPY --from=planner /app/target/release/deps /app/target/release/deps
COPY src ./src
COPY Cargo.toml Cargo.lock ./
# Build the application
RUN cargo build --release --bin server

# Stage 2: Create the final, small image
FROM debian:12-slim as runner
WORKDIR /app

# Create a non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 appuser
USER appuser

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/server .

# Copy the model checkpoints
# Ensure the user has permissions to read these files
USER root
COPY --chown=appuser:appuser checkpoints ./checkpoints
USER appuser

# Expose the port the app runs on
EXPOSE 3000

# Set the entrypoint to run the server
# The arguments will be passed via `docker run`
ENTRYPOINT ["./server"]
