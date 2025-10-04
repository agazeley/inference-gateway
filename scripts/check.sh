#!/bin/bash

# Script to run all CI steps locally

set -e

echo "Running formatting check..."
cargo fmt --all
cargo fmt --all -- --check

echo "Running linting..."
cargo clippy --all-targets --all-features -- -D warnings

echo "Running unit tests..."
cargo test --all

echo "Running build..."
cargo build --release

echo "Running audit..."
cargo audit

echo "All CI steps completed successfully."
