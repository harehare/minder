//! The host/plugin ABI: a hand-rolled "pack a (ptr, len) pair into an i64,
//! JSON in guest linear memory" convention -- deliberately not the wasm
//! component model, since this workspace has no component-model tooling and
//! this scheme is directly inspectable (no lifting/lowering indirection).
//!
//! A plugin exports:
//!   minder_alloc(len: i32) -> i32
//!   minder_dealloc(ptr: i32, len: i32)
//!   minder_tool_name() -> i64              (packed ptr:len, plain UTF-8 string)
//!   minder_tool_description() -> i64       (packed ptr:len, plain UTF-8 string)
//!   minder_tool_parameters_schema() -> i64  (packed ptr:len, UTF-8 JSON)
//!   minder_tool_execute(args_ptr: i32, args_len: i32) -> i64  (packed ptr:len, UTF-8 JSON)
//!
//! `minder_alloc`/`minder_dealloc` let the host allocate a buffer inside
//! guest memory to write call arguments into before invoking
//! `minder_tool_execute`, and let the host free buffers it reads results
//! from afterwards.

use wasmtime::{Memory, Store};
use wasmtime_wasi::p1::WasiP1Ctx;

/// Only ever produced by the guest; the host only ever unpacks. Kept here
/// (rather than duplicated in the test module) since it's the precise
/// inverse the guest side must implement -- documents the wire format.
#[cfg(test)]
fn pack(ptr: u32, len: u32) -> i64 {
    ((ptr as i64) << 32) | (len as i64 & 0xFFFF_FFFF)
}

pub fn unpack(packed: i64) -> (u32, u32) {
    let ptr = (packed >> 32) as u32;
    let len = (packed & 0xFFFF_FFFF) as u32;
    (ptr, len)
}

pub fn read_bytes(store: &Store<WasiP1Ctx>, memory: &Memory, ptr: u32, len: u32) -> Vec<u8> {
    let data = memory.data(store);
    let start = ptr as usize;
    let end = start.saturating_add(len as usize).min(data.len());
    if start >= data.len() || start >= end {
        return Vec::new();
    }
    data[start..end].to_vec()
}

pub fn read_packed_string(store: &Store<WasiP1Ctx>, memory: &Memory, packed: i64) -> String {
    let (ptr, len) = unpack(packed);
    String::from_utf8_lossy(&read_bytes(store, memory, ptr, len)).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let (ptr, len) = (12345u32, 678u32);
        let packed = pack(ptr, len);
        assert_eq!(unpack(packed), (ptr, len));
    }

    #[test]
    fn pack_unpack_handles_zero() {
        assert_eq!(unpack(pack(0, 0)), (0, 0));
    }
}
