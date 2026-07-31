//! Example plugin: runs an `mq` query against Markdown input. Pure
//! computation, no I/O -- needs zero manifest capabilities.

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
    write_string("mq")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string(
        "Runs an mq query (a jq-like query language for Markdown) against Markdown input and \
         returns the matched nodes/values, one per line.",
    )
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(
        r#"{"type":"object","properties":{"query":{"type":"string","description":"mq query, e.g. select(is_h1())"},"input":{"type":"string","description":"Markdown source to run the query against"}},"required":["query","input"]}"#,
    )
}

fn error_outcome(message: String) -> serde_json::Value {
    serde_json::json!({ "content": message, "is_error": true })
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(args_ptr: i32, args_len: i32) -> i64 {
    let raw = read_string(args_ptr, args_len);
    let args: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return write_string(&error_outcome(format!("invalid arguments: {e}")).to_string()),
    };

    let query = match args["query"].as_str() {
        Some(q) => q,
        None => return write_string(&error_outcome("missing required \"query\" argument".to_string()).to_string()),
    };
    let input = match args["input"].as_str() {
        Some(i) => i,
        None => return write_string(&error_outcome("missing required \"input\" argument".to_string()).to_string()),
    };

    let nodes = match mq_lang::parse_markdown_input(input) {
        Ok(nodes) => nodes,
        Err(e) => return write_string(&error_outcome(format!("invalid markdown input: {e}")).to_string()),
    };

    let mut engine = mq_lang::DefaultEngine::default();
    engine.load_builtin_module();

    let outcome = match engine.eval(query, nodes.into_iter()) {
        Ok(values) => {
            let content = values
                .into_iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            serde_json::json!({ "content": content, "is_error": false })
        }
        Err(e) => error_outcome(e.to_string()),
    };
    write_string(&outcome.to_string())
}
