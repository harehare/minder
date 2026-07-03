//! Test fixture: `minder_tool_execute` tries to read a path that no
//! `[[fs]]` capability in the manifest ever grants. Proves the sandbox
//! actually holds -- the plugin must see a WASI permission/lookup error
//! *itself*, not have the host specially intercept the attempt.

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
    write_string("fs_probe")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string("Test fixture: probes filesystem access outside any granted capability")
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(r#"{"type":"object"}"#)
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(_args_ptr: i32, _args_len: i32) -> i64 {
    let outcome = match std::fs::read_to_string("/ungranted/secret.txt") {
        Ok(contents) => serde_json::json!({ "content": contents, "is_error": false }),
        Err(e) => serde_json::json!({ "content": format!("denied: {e}"), "is_error": true }),
    };
    write_string(&outcome.to_string())
}
