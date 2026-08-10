---
name: changelog-entries
description: Writes a CHANGELOG.md entry summarizing the current diff
---
# Changelog entries

Look at the staged (or unstaged, if nothing is staged) diff and write one
`CHANGELOG.md` bullet under an `## Unreleased` heading (create it if
missing), in the same tense and format as the entries already there.

Summarize the user-visible effect of the change, not the diff line by line.
Skip changes with no user-visible effect (tests, internal refactors).
