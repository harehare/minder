//! Test fixture: `minder_tool_execute` never returns (tight infinite loop).
//! Proves the host's fuel/timeout enforcement actually aborts execution
//! within bounded wall-clock time rather than hanging forever.

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
    write_string("slow_loop")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string("Test fixture: loops forever")
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(r#"{"type":"object"}"#)
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(_args_ptr: i32, _args_len: i32) -> i64 {
    let mut x: u64 = 0;
    loop {
        x = std::hint::black_box(x.wrapping_add(1));
    }
}
