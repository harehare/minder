//! Test fixture: `minder_tool_execute` deliberately panics. With
//! `panic = "abort"` (set at the fixtures workspace level), this aborts the
//! wasm instance, which the host must see as a `wasmtime::Trap` and convert
//! to `ToolExecOutcome { is_error: true, .. }` -- not let it propagate as a
//! Rust panic in the host process.

use std::alloc::{Layout, alloc, dealloc};

fn main() {}

fn pack(ptr: u32, len: u32) -> i64 {
    ((ptr as i64) << 32) | (len as i64 & 0xFFFF_FFFF)
}

fn write_string(s: &str) -> i64 {
    let bytes = s.as_bytes();
    let ptr = minder_alloc(bytes.len() as i32);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
    }
    pack(ptr as u32, bytes.len() as u32)
}

#[no_mangle]
pub extern "C" fn minder_alloc(len: i32) -> i32 {
    let layout = Layout::from_size_align(len.max(1) as usize, 1).unwrap();
    unsafe { alloc(layout) as i32 }
}

#[no_mangle]
pub extern "C" fn minder_dealloc(ptr: i32, len: i32) {
    let layout = Layout::from_size_align(len.max(1) as usize, 1).unwrap();
    unsafe { dealloc(ptr as *mut u8, layout) }
}

#[no_mangle]
pub extern "C" fn minder_tool_name() -> i64 {
    write_string("panicking")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string("Test fixture: always panics")
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(r#"{"type":"object"}"#)
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(_args_ptr: i32, _args_len: i32) -> i64 {
    panic!("intentional test panic");
}
