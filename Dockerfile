# Multi-stage Dockerfile for ONNX test Rust project
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy over essentials
COPY ./src ./src
COPY ./.cargo ./.cargo
COPY ./Cargo.lock ./Cargo.toml ./

# Fetch then build to cache downloaded crates
RUN cargo fetch
RUN cargo build --release --verbose

# # Runtime stage with CUDA support
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04 AS model-based

# Create app user for security
RUN groupadd -r gateway && useradd -r -g gateway gateway

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /runtime

# Copy the built binary from builder stage
COPY --from=builder /build/target/release/inference-gateway /runtime/inference-gateway

# Copy the onnx libraries from the builder stage
COPY --from=builder /build/target/release/libonnx* /usr/local/lib/

# Change ownership to app user
RUN chown -R gateway:gateway /runtime

# Switch to non-root user
USER gateway

# Expose the default port
EXPOSE 3000

CMD ["/runtime/inference-gateway"]
