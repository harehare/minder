#!/usr/bin/env sh
# Regenerates the checked-in .wasm fixtures from source. Run manually after
# editing a fixture's src/main.rs -- these binaries are committed so the
# test suite doesn't need the wasm32-wasip1 target installed in CI.
#
# Prerequisite: rustup target add wasm32-wasip1
set -eu
cd "$(dirname "$0")"

cargo build --release --target wasm32-wasip1

for plugin in echo_plugin panicking_plugin slow_loop_plugin fs_probe_plugin net_probe_plugin; do
    cp "target/wasm32-wasip1/release/${plugin}.wasm" "${plugin}.wasm"
done

echo "regenerated: echo_plugin.wasm panicking_plugin.wasm slow_loop_plugin.wasm fs_probe_plugin.wasm"
