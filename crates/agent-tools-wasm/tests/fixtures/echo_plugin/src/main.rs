//! Test fixture: happy-path plugin. Echoes its arguments back in `content`,
//! proving the full ABI round trip (host writes JSON into guest memory via
//! `minder_alloc` -> guest reads it -> guest allocates+writes a response ->
//! host reads it back and frees both buffers via `minder_dealloc`).

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

fn read_string(ptr: i32, len: i32) -> String {
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    String::from_utf8_lossy(slice).into_owned()
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
    write_string("echo")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string("Test fixture: echoes back the arguments it was called with")
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(r#"{"type":"object"}"#)
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(args_ptr: i32, args_len: i32) -> i64 {
    let args = read_string(args_ptr, args_len);
    let outcome = serde_json::json!({ "content": args, "is_error": false });
    write_string(&outcome.to_string())
}
