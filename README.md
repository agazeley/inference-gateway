# Inference Gateway

A high-performance AI inference service built in Rust for production text generation workloads using ONNX models with persistent transaction storage.

## Overview

Inference Gateway provides a scalable HTTP API for text generation with enterprise-grade features including observability, health monitoring, configurable inference parameters, and persistent data storage. Built with Rust and ONNX Runtime for optimal performance and resource efficiency.

## Architecture

This project uses a **hybrid library + binary structure** optimized for both development and production deployment:

### Project Structure
- **`src/lib.rs`**: Core library (`inference_gateway`) with reusable components
- **`src/bin/server.rs`**: Main HTTP server binary (`inference-gateway`)
- **API Layer**: Axum-based HTTP service with JSON request/response handling
- **Inference Engine**: ONNX Runtime integration with autoregressive model support and KV-caching
- **Data Layer**: SQLite-based persistent storage with async transaction logging
- **Repository Pattern**: Abstracted database operations with trait-based architecture
- **Observability**: Prometheus metrics, structured logging, and distributed tracing
- **Concurrency**: Optimized database connection pooling and background transaction processing
- **Deployment**: Docker and Docker Compose support with multi-stage builds

## Features

### Core Functionality
- **Text Generation**: Configurable parameters (temperature, top-p, max tokens, min-p)
- **Model Support**: Multiple model formats via ONNX Runtime with dynamic shape handling
- **Prompt Templates**: Custom templating system with Jinja-based templates
- **Transaction Storage**: Persistent storage of inference requests and responses

### Database Features
- **SQLite Integration**: Embedded database for transaction persistence
- **Migration System**: Configurable database migrations (safe updates or full recreation)
- **Environment Configuration**: Database path and migration mode via environment variables

### Production Features
- **Docker Support**: Multi-stage Dockerfile with CUDA support
- **Container Orchestration**: Docker Compose configurations for development and production
- **Health Monitoring**: Health and readiness endpoints with container health checks
- **Observability**: Prometheus metrics, structured logging, and distributed tracing
- **Request Tracking**: Request ID propagation and correlation

## Installation

### Prerequisites
- **Rust**: 1.70+ (2024 edition)
- **ONNX Runtime**: Compatible with CUDA 12.9+ (optional for GPU acceleration)
- **Docker**: For containerized deployment (optional)

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd inference-gateway

# Production deployment
docker compose up --build -d

# Development deployment (with database reset and web interface)
docker compose -f docker-compose.dev.yml up --build
```

### Building and Running

```bash
# Build the library and server binary
cargo build --release

# Run the main server binary
cargo run --bin inference-gateway --release

# Or use the shorter form (since it's the default binary)
cargo run --release

# Run with custom database configuration
DATABASE_MIGRATION_MODE=create_if_not_exists cargo run --release

# Build only the library (for testing/development)
cargo build --lib

# Run library tests
cargo test --lib
```

The service starts on `0.0.0.0:3000` by default.

### Environment Configuration

Copy and customize the environment file:
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Configuration

Configure the service using environment variables:

### Core Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_DEFAULT_MODEL_NAME` | Model identifier | `gpt2` |
| `INFERENCE_DEFAULT_HF_MODEL_ID` | Hugging Face model ID for download | GPT-2 ONNX model URL |
| `RUST_LOG` | Logging configuration | `info,ort=warn,tower_http=info` |

### Database Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_CONNECTION_STRING_VAR` | Database connection string to use | `sqlite:file:$PWD/data/database.db?mode=rwc` |
| `DATABASE_MIGRATION_MODE` | Migration strategy | `create_if_not_exists` |

### Migration Modes
- **`create_if_not_exists`**: Safe mode - preserves existing data, creates tables if missing
- **`drop_recreate`**: Development mode - **DELETES ALL DATA** and recreates tables

### Docker Configuration
The Docker Compose setup supports additional environment variables for container orchestration:

```yaml
# Production mode (safe migrations)
DATABASE_MIGRATION_MODE=create_if_not_exists

# Development mode (reset database on startup)
DATABASE_MIGRATION_MODE=drop_recreate
```

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
  "model": "mymodel",
  "text": "The future of artificial intelligence looks promising with advances in machine learning...",
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
- `min_p` (optional): Minimum probability threshold (0.0-1.0, default: 0.0)

### Transaction Management

**Endpoint:** `GET /transactions`
- Retrieve all stored inference transactions

**Endpoint:** `POST /transactions`
- Store a new inference transaction

**Request Body:**
```json
{
  "transaction": {
    "prompt": "User input text",
    "response": "Generated response",
    "model_name": "model-identifier"
  }
}
```

### Health Endpoints

- `GET /healthz` - Service health check (always returns 200 when service is running)
- `GET /readyz` - Readiness check (returns 200 when model is loaded and ready)
- `GET /metrics` - Prometheus metrics in OpenMetrics format

### Database Web Interface (Development)
When using the development Docker Compose setup, a SQLite web interface is available:
- **URL**: http://localhost:8080
- **Features**: Browse, query, and manage SQLite database content

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

### Local Development Setup
```bash
# Formating
cargo fmt

# Linting
cargo check
cargo clippy --fix --allow-dirty

# Testing (library and binary)
cargo test

# Integration tests
pytest -n 4 tests/integration
```

```bash
# Install development dependencies
cargo install cargo-watch cargo-nextest

# Run with auto-reload
cargo watch -x "clippy --profile test" -x "test" -x run

# Run with database reset (development mode)
DATABASE_MIGRATION_MODE=drop_recreate cargo run
```

### Docker Development
```bash
# Development environment with database viewer
docker compose -f docker-compose.dev.yml up --build

# Access services:
# - API: http://localhost:3000
# - Database viewer: http://localhost:8080

# Reset development environment
docker compose -f docker-compose.dev.yml down -v
docker compose -f docker-compose.dev.yml up --build
```

### Testing
```bash
# Run all tests
cargo test

# Run with nextest (faster parallel execution)
cargo nextest run

# Integration tests
cargo test --test integration

# Test with clean database
DATABASE_MIGRATION_MODE=drop_recreate cargo test
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

### Database Operations
```bash
# Safe migration (preserves data)
DATABASE_MIGRATION_MODE=create_if_not_exists cargo run

# Reset database (deletes all data)
DATABASE_MIGRATION_MODE=drop_recreate cargo run

# Custom database path
DATABASE_CONNECTION_STRING_VAR=sqlite://./custom.db cargo run
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
