### Fixed: the MuJoCo tool_spec description no longer advertises an action the schema omits

`tool_spec`'s description enumerates the action surface as `Actions (77 total):
[Category] ...`, and that sentence listed 78 names. The surplus one was
`save_episode`, under `[Recording]` beside `start_recording` and
`stop_recording`, while the `action` enum did not offer it - so a model
following the description would emit a value a schema-constrained decoder
cannot select, and the documented call came back rejected.

Dropping the name rather than publishing it is what the surrounding state
already decided: `save_episode` is declared deliberately Python-only in the
dispatchable-but-unadvertised inventory, and `run_policy(n_episodes=N)` is the
published multi-episode path that flushes a boundary per episode. Nothing an
agent could reach was lost, and no Python caller is affected - the router still
resolves `save_episode` by name.

Both existing drift checks passed throughout, because both read the enum: one
asks whether each *enum* entry appears in the text, so a surplus name is not a
value it iterates, and the other compared the stated `77` against `len(enum)`,
which agreed. The count literal was the one signal the lists had diverged, and
it was measured against the enum rather than against the list it introduces.
The converse direction is now pinned - every name the category lists carry must
be in the enum, and the stated count must match the number of names listed.
