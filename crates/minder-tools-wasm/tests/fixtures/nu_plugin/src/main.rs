//! Example plugin: runs a Nushell script/pipeline entirely inside the
//! sandbox. `default-features = false` drops the `os` feature (crossterm,
//! reedline, sysinfo, os_pipe, notify -- none of which compile for
//! `wasm32-wasip1`), leaving pure data-pipeline commands only. Needs zero
//! manifest capabilities.

use std::alloc::{Layout, alloc, dealloc};

use nu_cmd_lang::create_default_context;
use nu_command::add_shell_command_context;
use nu_engine::eval_block;
use nu_parser::parse;
use nu_protocol::debugger::WithoutDebug;
use nu_protocol::engine::{Stack, StateWorkingSet};
use nu_protocol::{Config, PipelineData, Span, Value};

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
    write_string("nu")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string(
        "Runs a Nushell script against an optional string input and returns the resulting \
         value. No filesystem, process, or network commands are available -- only the pure \
         data pipeline (filters, strings, math, formats).",
    )
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(
        r#"{"type":"object","properties":{"script":{"type":"string","description":"Nushell script, e.g. \"$in | each { |x| $x * 2 } | to json --raw\""},"input":{"type":"string","description":"Optional string piped in as $in"}},"required":["script"]}"#,
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

    let script = match args["script"].as_str() {
        Some(s) => s,
        None => return write_string(&error_outcome("missing required \"script\" argument".to_string()).to_string()),
    };

    let mut engine_state = add_shell_command_context(create_default_context());

    let mut working_set = StateWorkingSet::new(&engine_state);
    let block = parse(&mut working_set, None, script.as_bytes(), false);
    if let Some(err) = working_set.parse_errors.first() {
        return write_string(&error_outcome(format!("parse error: {err}")).to_string());
    }
    let delta = working_set.render();

    if let Err(e) = engine_state.merge_delta(delta) {
        return write_string(&error_outcome(format!("failed to merge parsed script: {e}")).to_string());
    }

    let input = match args["input"].as_str() {
        Some(s) => PipelineData::Value(Value::string(s, Span::unknown()), None),
        None => PipelineData::empty(),
    };

    let mut stack = Stack::new();
    let outcome = match eval_block::<WithoutDebug>(&engine_state, &mut stack, &block, input) {
        Ok(data) => match data.body.into_value(Span::unknown()) {
            Ok(value) => {
                let content = value.to_expanded_string("\n", &Config::default());
                serde_json::json!({ "content": content, "is_error": false })
            }
            Err(e) => error_outcome(e.to_string()),
        },
        Err(e) => error_outcome(e.to_string()),
    };
    write_string(&outcome.to_string())
}
