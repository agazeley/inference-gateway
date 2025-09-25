# Inference Gateway

A high-performance AI inference service built in Rust for production text generation workloads using ONNX models.

## Overview

Inference Gateway provides a scalable HTTP API for text generation with enterprise-grade features including observability, health monitoring, and configurable inference parameters. Built with Rust and ONNX Runtime for optimal performance and resource efficiency.

## Architecture

- **API Layer**: Axum-based HTTP service with JSON request/response handling
- **Inference Engine**: ONNX Runtime integration with autoregressive model support
- **Observability**: Prometheus metrics, structured logging, and distributed tracing
- **Concurrency**: Thread-safe model sharing with async request handling

## Features

### Core Functionality
- Text generation with configurable parameters (temperature, top-p, max tokens)
- Support for multiple model formats via ONNX Runtime
- Batched inference capabilities
- Custom prompt templating system

### Production Features
- Prometheus metrics collection
- Structured logging with configurable levels
- Health and readiness endpoints
- Request ID propagation
- Graceful shutdown handling

## Installation

### Prerequisites
- Rust 1.70+ (2024 edition)
- ONNX Runtime dependencies

### Building from Source
```bash
git clone <repository-url>
cd inference-gateway
cargo build --release
```

### Running
```bash
cargo run --release
```

The service starts on `0.0.0.0:3000` by default.

## Configuration

Configure the service using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_DEFAULT_MODEL_NAME` | Model identifier | `gpt2` |
| `INFERENCE_DEFAULT_MODEL_PATH` | Model URL or local path | GPT-2 ONNX model URL |
| `INFERENCE_DEFAULT_TOKENIZER` | Tokenizer identifier | `openai-community/gpt2` |
| `RUST_LOG` | Logging configuration | `info,ort=warn,tower_http=info` |

## API Reference

### Generate Text

**Endpoint:** `POST /api/v1/inference`

**Request Body:**
```json
{
  "text": "The future of artificial intelligence",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "generated_text": "The future of artificial intelligence looks promising with advances in machine learning...",
  "metadata": {
    "tokens_generated": 50,
    "processing_time_ms": 245
  }
}
```

**Parameters:**
- `text` (required): Input prompt for text generation
- `max_tokens` (optional): Maximum number of tokens to generate (default: 50)
- `temperature` (optional): Controls randomness (0.0-2.0, default: 1.0)
- `top_p` (optional): Nucleus sampling parameter (0.0-1.0, default: 1.0)

### Health Endpoints

- `GET /healthz` - Service health check (always returns 200 when service is running)
- `GET /readyz` - Readiness check (returns 200 when model is loaded and ready)
- `GET /metrics` - Prometheus metrics in OpenMetrics format

## Monitoring

### Metrics

The service exposes Prometheus metrics on `/metrics`:

- `inference_gateway_http_requests_total` - Total HTTP requests by method and status
- `inference_gateway_http_request_duration_seconds` - Request duration histogram
- `inference_gateway_inference_duration_seconds` - Model inference duration
- `inference_gateway_active_connections` - Current active connections

### Logging

Structured JSON logs with configurable levels. Key log events:
- Request/response logging with correlation IDs
- Model loading and initialization
- Error conditions and performance metrics

Example log configuration:
```bash
RUST_LOG=info,inference_gateway=debug,tower_http=info cargo run
```

## Development

### Setup
```bash
# Install development dependencies
cargo install cargo-watch cargo-nextest

# Run with auto-reload
cargo watch -x "clippy --profile test" -x "test" -x run
```

### Testing
```bash
# Run all tests
cargo test

# Run with nextest (faster parallel execution)
cargo nextest run

# Integration tests
cargo test --test integration
```

### Code Quality
```bash
# Format code
cargo fmt

# Lint code
cargo clippy --profile test

# Security audit
cargo audit
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`cargo test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

[Add your license here]

## Support

For issues, questions, or contributions, please open an issue on the repository.