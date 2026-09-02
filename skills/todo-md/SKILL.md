---
name: todo-md
description: Tracks multi-step task progress in TODO.md instead of an in-session todo list
---
# TODO.md task tracking

For any task with several non-trivial steps, keep a checklist in `TODO.md`
at the project root instead of only holding the plan in your own reasoning.

1. Before starting, use `write_file`/`edit_file` to lay out the steps as GFM
   checkboxes: `- [ ] step`.
2. Mark a step `- [x]` as soon as it's genuinely done, not preemptively.
3. Re-read `TODO.md` with `read_file` before continuing work across turns so
   the plan survives compaction.

This is the same checkbox format `minder loop TODO.md` consumes, so a
checklist built this way can be handed to unattended `loop` mode later
without reformatting.
