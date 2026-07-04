---
name: commit-messages
description: Writes commit messages in this repo's conventional-commit style
---
# Commit messages

Use Conventional Commits: `<type>(<scope>): <summary>`, imperative mood, no
trailing period, summary under 72 characters.

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `chore`.

Before writing the message:

1. Run `git diff --staged` (or `git diff` if nothing is staged) to see what
   actually changed.
2. Pick the type that matches the *dominant* change -- if a commit both fixes
   a bug and adds a test, prefer `fix`.
3. Keep the scope to a single crate or module name when the change is
   localized (e.g. `fix(agent-tools): ...`); omit the scope for changes that
   span the workspace.
4. Do not describe *what* the diff shows line by line -- summarize the
   effect of the change.
