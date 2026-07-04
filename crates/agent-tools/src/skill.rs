use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// A project skill: a named, on-demand instruction set the model can pull
/// into context via `SkillTool` instead of carrying every skill's full body
/// in the system prompt on every turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SkillLoadError {
    #[error("failed to read skills directory {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("skill file {0} is missing '---' delimited frontmatter")]
    MissingFrontmatter(PathBuf),
    #[error("skill file {0} frontmatter is missing required field '{1}'")]
    MissingField(PathBuf, &'static str),
    #[error("duplicate skill name '{name}' in {first} and {second} -- skill names must be unique")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
}

/// Discovers skills from `agent_dir/skills/*/SKILL.md`, one directory per
/// skill (Claude-Code-style skill packages). Returns an empty vec if the
/// skills directory doesn't exist -- skills are fully optional, like hooks
/// and wasm tool plugins.
pub fn discover_skills(agent_dir: &Path) -> Result<Vec<Skill>, SkillLoadError> {
    let skills_dir = agent_dir.join("skills");
    if !skills_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&skills_dir)
        .map_err(|e| SkillLoadError::Io(skills_dir.clone(), e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    entries.sort();

    let mut skills: Vec<Skill> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new(); // parallel to `skills`, for error messages
    for dir in entries {
        let skill_md = dir.join("SKILL.md");
        if !skill_md.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&skill_md)
            .map_err(|e| SkillLoadError::Io(skill_md.clone(), e))?;
        let skill = parse_skill(&skill_md, &raw)?;

        if let Some(idx) = skills.iter().position(|s| s.name == skill.name) {
            return Err(SkillLoadError::DuplicateName {
                name: skill.name,
                first: sources[idx].clone(),
                second: skill_md,
            });
        }
        sources.push(skill_md);
        skills.push(skill);
    }

    Ok(skills)
}

fn parse_skill(path: &Path, raw: &str) -> Result<Skill, SkillLoadError> {
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw); // tolerate a UTF-8 BOM
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
        .ok_or_else(|| SkillLoadError::MissingFrontmatter(path.to_path_buf()))?;
    let end = rest
        .find("\n---")
        .ok_or_else(|| SkillLoadError::MissingFrontmatter(path.to_path_buf()))?;
    let frontmatter = &rest[..end];
    let body = rest[end..]
        .trim_start_matches("\n---")
        .trim_start_matches("\r\n---")
        .trim_start_matches(['\r', '\n']);

    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key.trim() {
            "name" => name = Some(value.to_string()),
            "description" => description = Some(value.to_string()),
            _ => {}
        }
    }

    let name = name.ok_or_else(|| SkillLoadError::MissingField(path.to_path_buf(), "name"))?;
    let description = description
        .ok_or_else(|| SkillLoadError::MissingField(path.to_path_buf(), "description"))?;

    Ok(Skill {
        name,
        description,
        body: body.trim().to_string(),
    })
}

/// Exposes discovered skills to the model as a single `skill` tool: the
/// tool description lists every skill's name and short description (cheap,
/// always in context), and calling it with a `name` loads that skill's
/// full body into the conversation (only paid for on demand).
pub struct SkillTool {
    skills: Vec<Skill>,
    description: String,
}

impl SkillTool {
    pub fn new(skills: Vec<Skill>) -> Self {
        let description = if skills.is_empty() {
            "Loads the full instructions for a named project skill.".to_string()
        } else {
            let list = skills
                .iter()
                .map(|s| format!("- {}: {}", s.name, s.description))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "Loads the full instructions for a named project skill. Call this before \
                 attempting a task that matches one of the skills below -- the short \
                 description alone is not enough to act on, load the skill body first.\n\n\
                 Available skills:\n{list}"
            )
        };
        Self {
            skills,
            description,
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        "skill"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let names: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to load",
                    "enum": names
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let name = match arguments.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return error("missing required argument 'name'".to_string()),
        };

        match self.skills.iter().find(|s| s.name == name) {
            Some(skill) => ToolExecOutcome {
                content: skill.body.clone(),
                is_error: false,
                metadata: serde_json::Value::Null,
            },
            None => error(format!(
                "unknown skill '{name}' -- available: {}",
                self.skills
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

fn error(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-skill-test-{}", uuid::Uuid::new_v4()))
    }

    fn write_skill(agent_dir: &Path, dir_name: &str, contents: &str) {
        let skill_dir = agent_dir.join("skills").join(dir_name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), contents).unwrap();
    }

    #[test]
    fn discovers_no_skills_when_skills_dir_is_absent() {
        let agent_dir = scratch_dir();
        let skills = discover_skills(&agent_dir).unwrap();
        assert!(skills.is_empty());
    }

    #[test]
    fn discovers_and_parses_a_skill() {
        let agent_dir = scratch_dir();
        write_skill(
            &agent_dir,
            "commit-messages",
            "---\nname: commit-messages\ndescription: Writes conventional commit messages\n---\n# Commit messages\n\nUse the conventional commits format.\n",
        );

        let skills = discover_skills(&agent_dir).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "commit-messages");
        assert_eq!(skills[0].description, "Writes conventional commit messages");
        assert_eq!(
            skills[0].body,
            "# Commit messages\n\nUse the conventional commits format."
        );
    }

    #[test]
    fn skips_directories_without_a_skill_md() {
        let agent_dir = scratch_dir();
        std::fs::create_dir_all(agent_dir.join("skills").join("not-a-skill")).unwrap();
        let skills = discover_skills(&agent_dir).unwrap();
        assert!(skills.is_empty());
    }

    #[test]
    fn missing_frontmatter_is_an_error() {
        let agent_dir = scratch_dir();
        write_skill(&agent_dir, "bad", "# no frontmatter here\n");
        let err = discover_skills(&agent_dir).unwrap_err();
        assert!(matches!(err, SkillLoadError::MissingFrontmatter(_)));
    }

    #[test]
    fn missing_required_field_is_an_error() {
        let agent_dir = scratch_dir();
        write_skill(&agent_dir, "bad", "---\nname: bad\n---\nbody\n");
        let err = discover_skills(&agent_dir).unwrap_err();
        assert!(matches!(
            err,
            SkillLoadError::MissingField(_, "description")
        ));
    }

    #[test]
    fn duplicate_skill_names_are_an_error() {
        let agent_dir = scratch_dir();
        write_skill(
            &agent_dir,
            "a",
            "---\nname: dup\ndescription: first\n---\nbody\n",
        );
        write_skill(
            &agent_dir,
            "b",
            "---\nname: dup\ndescription: second\n---\nbody\n",
        );
        let err = discover_skills(&agent_dir).unwrap_err();
        assert!(matches!(err, SkillLoadError::DuplicateName { name, .. } if name == "dup"));
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn loads_a_skill_body_by_name() {
        let tool = SkillTool::new(vec![Skill {
            name: "example".to_string(),
            description: "an example skill".to_string(),
            body: "full instructions here".to_string(),
        }]);
        let outcome = tool
            .execute(serde_json::json!({"name": "example"}), &ctx())
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "full instructions here");
    }

    #[tokio::test]
    async fn unknown_skill_name_is_an_error() {
        let tool = SkillTool::new(vec![]);
        let outcome = tool
            .execute(serde_json::json!({"name": "nope"}), &ctx())
            .await;
        assert!(outcome.is_error);
    }

    #[test]
    fn tool_description_lists_available_skills() {
        let tool = SkillTool::new(vec![Skill {
            name: "example".to_string(),
            description: "an example skill".to_string(),
            body: String::new(),
        }]);
        assert!(tool.description().contains("example: an example skill"));
    }
}
