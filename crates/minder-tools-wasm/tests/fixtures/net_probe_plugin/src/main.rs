//! Test fixture: `minder_tool_execute` calls the host-mediated
//! `host_web_fetch` import (only resolvable when the manifest grants
//! `network = true`) and echoes back what it got.

use std::alloc::{Layout, alloc, dealloc};

const OUT_CAP: i32 = 65536;

#[link(wasm_import_module = "minder")]
unsafe extern "C" {
    fn host_web_fetch(url_ptr: i32, url_len: i32, out_ptr: i32, out_cap: i32) -> i32;
}

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
    write_string("net_probe")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string("Test fixture: calls host_web_fetch")
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(r#"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"#)
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(args_ptr: i32, args_len: i32) -> i64 {
    let args_raw = read_string(args_ptr, args_len);
    let args: serde_json::Value = serde_json::from_str(&args_raw).unwrap_or_default();
    let url = args["url"].as_str().unwrap_or("");

    let url_ptr = minder_alloc(url.len() as i32);
    unsafe {
        std::ptr::copy_nonoverlapping(url.as_ptr(), url_ptr as *mut u8, url.len());
    }
    let out_ptr = minder_alloc(OUT_CAP);

    let n = unsafe { host_web_fetch(url_ptr, url.len() as i32, out_ptr, OUT_CAP) };
    minder_dealloc(url_ptr, url.len() as i32);

    let is_error = n < 0;
    let written = n.unsigned_abs() as i32;
    let payload = read_string(out_ptr, written);
    minder_dealloc(out_ptr, OUT_CAP);

    let outcome = if is_error {
        serde_json::json!({ "content": payload, "is_error": true })
    } else {
        serde_json::json!({ "content": payload, "is_error": false })
    };
    write_string(&outcome.to_string())
}
