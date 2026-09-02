#!/usr/bin/env sh
# Regenerates ../todo.wasm from source. Run manually after editing src/main.rs.
#
# Prerequisite: rustup target add wasm32-wasip1
set -eu
cd "$(dirname "$0")"

cargo build --release --target wasm32-wasip1
cp target/wasm32-wasip1/release/todo_plugin.wasm ../todo.wasm

echo "regenerated: ../todo.wasm"
