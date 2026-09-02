//! `todo_write` as a WASM tool plugin, state persisted via the granted `[[fs]]` dir.

use std::alloc::{alloc, dealloc, Layout};
use std::io::ErrorKind;

use serde::{Deserialize, Serialize};

const STATE_PATH: &str = "/data/todo-state.json";

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
    write_string("todo_write")
}

#[no_mangle]
pub extern "C" fn minder_tool_description() -> i64 {
    write_string(
        "Replaces the current todo list with a full, updated list -- use it to plan and track \
         progress on a multi-step task. Always pass the *entire* list, not just the items that \
         changed. Keep at most one item `in_progress` at a time, and mark an item `completed` as \
         soon as it's actually done rather than preemptively. Persists to disk, so the list \
         survives across sessions.",
    )
}

#[no_mangle]
pub extern "C" fn minder_tool_parameters_schema() -> i64 {
    write_string(
        r#"{"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string","description":"Short description of the step"},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","status"]}}},"required":["todos"]}"#,
    )
}

#[derive(Serialize, Deserialize, Clone)]
struct TodoItem {
    content: String,
    status: String,
}

#[derive(Deserialize)]
struct Args {
    todos: Vec<TodoItem>,
}

fn format_checklist(todos: &[TodoItem]) -> String {
    if todos.is_empty() {
        return "(empty)".to_string();
    }
    todos
        .iter()
        .map(|t| {
            let mark = match t.status.as_str() {
                "pending" => "☐",
                "in_progress" => "◐",
                "completed" => "☑",
                _ => "?",
            };
            format!("{mark} {}", t.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[no_mangle]
pub extern "C" fn minder_tool_execute(args_ptr: i32, args_len: i32) -> i64 {
    let raw = read_string(args_ptr, args_len);
    let args: Args = match serde_json::from_str(&raw) {
        Ok(a) => a,
        Err(e) => {
            let outcome = serde_json::json!({ "content": format!("invalid arguments: {e}"), "is_error": true });
            return write_string(&outcome.to_string());
        }
    };

    if let Err(e) = std::fs::write(STATE_PATH, serde_json::to_string(&args.todos).unwrap_or_default()) {
        let hint = if e.kind() == ErrorKind::NotFound {
            " -- is the `[[fs]]` capability in todo.toml granted?"
        } else {
            ""
        };
        let outcome =
            serde_json::json!({ "content": format!("failed to persist todo list: {e}{hint}"), "is_error": true });
        return write_string(&outcome.to_string());
    }

    let pending = args.todos.iter().filter(|t| t.status == "pending").count();
    let in_progress = args.todos.iter().filter(|t| t.status == "in_progress").count();
    let completed = args.todos.iter().filter(|t| t.status == "completed").count();

    let outcome = serde_json::json!({
        "content": format_checklist(&args.todos),
        "is_error": false,
        "metadata": {
            "todos": args.todos.iter().map(|t| serde_json::json!({"content": t.content, "status": t.status})).collect::<Vec<_>>(),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
        }
    });
    write_string(&outcome.to_string())
}
