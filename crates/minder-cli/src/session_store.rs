use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use minder_core::{Message, Role};
use serde::{Deserialize, Serialize};

const SUMMARY_CHARS: usize = 80;

/// A saved conversation transcript, persisted under `.agent/sessions/` in
/// the project so `--continue`/`--resume` can pick it back up in a later
/// process. Kept CLI-side (not in minder-core) since core has no file I/O.
#[derive(Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub summary: String,
}

impl SessionRecord {
    /// `system_prompt`/`messages` start empty and are filled in by the
    /// first `save` call, once the session has actually run.
    pub fn new() -> Self {
        Self::with_id(uuid::Uuid::new_v4().to_string())
    }

    /// Like `new`, but with a caller-chosen id -- e.g. `minder loop`'s
    /// deterministic id (see `key_for_path`) so re-running the same
    /// checklist always resumes the same session file.
    pub fn with_id(id: String) -> Self {
        let now = unix_now();
        Self {
            id,
            created_at: now,
            updated_at: now,
            system_prompt: String::new(),
            messages: Vec::new(),
            summary: String::new(),
        }
    }
}

/// A stable, filesystem-safe session id derived from a file's canonical
/// path -- lets `minder loop <file>` resume the same session across
/// process restarts without the caller having to track a session id.
pub fn key_for_path(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let sanitized: String = canonical
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("loop-{sanitized}")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn sessions_dir(working_dir: &Path) -> PathBuf {
    working_dir.join(".agent").join("sessions")
}

fn path_for(working_dir: &Path, id: &str) -> PathBuf {
    sessions_dir(working_dir).join(format!("{id}.json"))
}

fn summarize(messages: &[Message]) -> String {
    let Some(first_user) = messages.iter().find(|m| m.role == Role::User) else {
        return String::new();
    };
    let text = first_user.text();
    let text = text.trim();
    let mut summary: String = text.chars().take(SUMMARY_CHARS).collect();
    if text.chars().count() > SUMMARY_CHARS {
        summary.push_str("...");
    }
    summary.replace('\n', " ")
}

/// Saves (creating or overwriting) a session, refreshing `updated_at` and
/// `summary`. Drops a `.gitignore` next to the transcripts on first use so
/// they aren't accidentally committed.
pub fn save(working_dir: &Path, record: &mut SessionRecord) -> io::Result<()> {
    record.updated_at = unix_now();
    record.summary = summarize(&record.messages);

    let dir = sessions_dir(working_dir);
    fs::create_dir_all(&dir)?;
    let gitignore = dir.join(".gitignore");
    if !gitignore.exists() {
        fs::write(&gitignore, "*\n!.gitignore\n")?;
    }

    let json = serde_json::to_vec_pretty(record).map_err(io::Error::other)?;
    fs::write(path_for(working_dir, &record.id), json)
}

fn read_record(path: &Path) -> io::Result<SessionRecord> {
    let data = fs::read(path)?;
    serde_json::from_slice(&data).map_err(io::Error::other)
}

fn json_files(working_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let dir = sessions_dir(working_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            files.push(path);
        }
    }
    Ok(files)
}

/// Loads a session by exact id, or by unambiguous id prefix.
pub fn load_by_id(working_dir: &Path, id: &str) -> io::Result<Option<SessionRecord>> {
    let exact = path_for(working_dir, id);
    if exact.exists() {
        return Ok(Some(read_record(&exact)?));
    }

    let mut matches: Vec<PathBuf> = json_files(working_dir)?
        .into_iter()
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with(id))
        })
        .collect();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(read_record(&matches.remove(0))?)),
        n => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("session id '{id}' is ambiguous ({n} matches)"),
        )),
    }
}

/// Loads the most recently updated session for this project, if any.
pub fn load_latest(working_dir: &Path) -> io::Result<Option<SessionRecord>> {
    let mut best: Option<SessionRecord> = None;
    for path in json_files(working_dir)? {
        let record = read_record(&path)?;
        if best.as_ref().is_none_or(|b| record.updated_at > b.updated_at) {
            best = Some(record);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::Message;

    fn scratch_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("minder-session-store-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_then_load_by_exact_id_round_trips() {
        let dir = scratch_dir();
        let mut record = SessionRecord::new();
        record.system_prompt = "sys".to_string();
        record.messages = vec![Message::user_text("hello there")];
        save(&dir, &mut record).unwrap();

        let loaded = load_by_id(&dir, &record.id).unwrap().unwrap();
        assert_eq!(loaded.system_prompt, "sys");
        assert_eq!(loaded.summary, "hello there");
    }

    #[test]
    fn load_by_id_matches_unambiguous_prefix() {
        let dir = scratch_dir();
        let mut record = SessionRecord::new();
        save(&dir, &mut record).unwrap();

        let prefix = &record.id[..8];
        let loaded = load_by_id(&dir, prefix).unwrap().unwrap();
        assert_eq!(loaded.id, record.id);
    }

    /// Writes a record with an explicit `updated_at`, bypassing `save`'s
    /// real-clock timestamp so ordering can be tested deterministically.
    fn write_with_timestamp(dir: &Path, mut record: SessionRecord, updated_at: u64) {
        record.updated_at = updated_at;
        fs::create_dir_all(sessions_dir(dir)).unwrap();
        let json = serde_json::to_vec_pretty(&record).unwrap();
        fs::write(path_for(dir, &record.id), json).unwrap();
    }

    #[test]
    fn load_latest_picks_the_most_recently_saved() {
        let dir = scratch_dir();
        write_with_timestamp(&dir, SessionRecord::new(), 100);
        let newer = SessionRecord::new();
        let newer_id = newer.id.clone();
        write_with_timestamp(&dir, newer, 200);

        let latest = load_latest(&dir).unwrap().unwrap();
        assert_eq!(latest.id, newer_id);
    }

    #[test]
    fn load_by_id_returns_none_when_missing() {
        let dir = scratch_dir();
        assert!(load_by_id(&dir, "does-not-exist").unwrap().is_none());
    }

    #[test]
    fn key_for_path_is_stable_and_distinguishes_different_paths() {
        let dir = scratch_dir();
        let a = dir.join("TODO.md");
        let b = dir.join("OTHER.md");
        std::fs::write(&a, "").unwrap();
        std::fs::write(&b, "").unwrap();

        assert_eq!(key_for_path(&a), key_for_path(&a));
        assert_ne!(key_for_path(&a), key_for_path(&b));
    }

    #[test]
    fn loop_session_round_trips_by_its_deterministic_key() {
        let dir = scratch_dir();
        let checklist = dir.join("TODO.md");
        std::fs::write(&checklist, "- [ ] a\n").unwrap();

        let key = key_for_path(&checklist);
        let mut record = SessionRecord::with_id(key.clone());
        record.messages = vec![Message::user_text("work on it")];
        save(&dir, &mut record).unwrap();

        let loaded = load_by_id(&dir, &key).unwrap().unwrap();
        assert_eq!(loaded.id, key);
    }
}
