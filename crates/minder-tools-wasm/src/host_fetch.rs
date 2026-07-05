//! `host_web_fetch`: the one network primitive a plugin can be granted,
//! gated by the manifest's `network = true` capability. Reuses the exact
//! same SSRF/timeout-guarded path as the built-in `web_fetch` tool
//! (`minder_tools::fetch`) -- not a reimplementation.
//!
//! ABI (avoids reentrant guest calls -- the guest pre-allocates its own
//! output buffer and passes its capacity, rather than the host calling back
//! into the guest's `minder_alloc`):
//!
//!   host_web_fetch(url_ptr: i32, url_len: i32, out_ptr: i32, out_cap: i32) -> i32
//!
//! On success, returns the number of bytes (>= 0) written to `out_ptr`,
//! encoding UTF-8 JSON `{"status":u16,"body":string,"truncated":bool}`
//! (truncated further if it doesn't fit `out_cap`, independent of the
//! guard's own `max_bytes` truncation). On failure, returns `-N` where `N`
//! is the number of bytes of a UTF-8 error message written to `out_ptr`.

use std::time::Duration;
use wasmtime::{AsContext, Caller, Linker};
use wasmtime_wasi::p1::WasiP1Ctx;

const HOST_FETCH_MAX_BYTES: usize = 1_000_000;
const HOST_FETCH_TIMEOUT_SECS: u64 = 30;

/// Links `host_web_fetch` into `linker` under the `minder` module name.
/// Callers only do this when the plugin's manifest grants `network = true`
/// -- a plugin that imports this function without the capability simply
/// fails to instantiate (unresolved import), which is inherently safe.
pub fn link(linker: &mut Linker<WasiP1Ctx>) -> Result<(), wasmtime::Error> {
    linker.func_wrap_async(
        "minder",
        "host_web_fetch",
        |mut caller: Caller<'_, WasiP1Ctx>, (url_ptr, url_len, out_ptr, out_cap): (i32, i32, i32, i32)| {
            Box::new(async move {
                let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                    Some(m) => m,
                    None => return -1i32,
                };

                let url_bytes = read_bytes(&caller, &memory, url_ptr, url_len);
                let url = String::from_utf8_lossy(&url_bytes).into_owned();

                let client = reqwest::Client::new();
                let timeout = Duration::from_secs(HOST_FETCH_TIMEOUT_SECS);
                let (payload, is_error) = match minder_tools::fetch(&client, &url, HOST_FETCH_MAX_BYTES, timeout).await
                {
                    Ok(result) => (
                        serde_json::json!({
                            "status": result.status,
                            "body": result.body,
                            "truncated": result.truncated,
                        })
                        .to_string(),
                        false,
                    ),
                    Err(e) => (e, true),
                };

                let bytes = payload.as_bytes();
                let n = bytes.len().min(out_cap.max(0) as usize);
                if memory.write(&mut caller, out_ptr as usize, &bytes[..n]).is_err() {
                    return -1i32;
                }

                if is_error { -(n as i32) } else { n as i32 }
            })
        },
    )?;
    Ok(())
}

fn read_bytes(caller: &Caller<'_, WasiP1Ctx>, memory: &wasmtime::Memory, ptr: i32, len: i32) -> Vec<u8> {
    let data = memory.data(caller.as_context());
    let start = ptr.max(0) as usize;
    let end = start.saturating_add(len.max(0) as usize).min(data.len());
    if start >= data.len() || start >= end {
        return Vec::new();
    }
    data[start..end].to_vec()
}
