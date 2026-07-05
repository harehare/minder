set working-directory := '.'

export RUST_BACKTRACE := "1"

# Run the CLI with the provided arguments
run *args:
    cargo run -p agent-cli -- {{args}}

# Build the project in release mode
build:
    cargo build --release --workspace

# Run formatting check
fmt:
    cargo fmt --all -- --check

# Run doc tests
test-doc:
    cargo test --doc --workspace --all-features

# Run all tests
test:
    cargo nextest run --workspace --all-features

# Run formatting, linting and all tests
test-all: fmt lint test-doc test

# Run tests with code coverage reporting
test-cov:
    cargo llvm-cov --open --html --workspace --all-features

# Run the linter
lint:
    cargo clippy --all-targets --all-features --workspace -- -D clippy::all

# Check for unused dependencies
deps:
    cargo machete

# Run cargo-deny (license/bans/sources/advisories)
audit:
    cargo deny check
