### Changed: the `strands-agents` floor rises to 1.13.0, the release that makes `BeforeToolCallEvent` usable

`strands.hooks.BeforeToolCallEvent` is imported at module scope by
`strands_robots.dashboard.agent_hitl`, the human-in-the-loop gate that pauses an
agent tool call before real hardware moves. Measured against every released 1.x
wheel, the name and the API the gate uses on it do not ship together:

- **1.10.0** exports the class from `strands.hooks`. Below that the name is
  absent rather than moved, so on 1.9.1 -- a release the old `>=1.7.0` floor
  admitted -- importing that module raises `ImportError: cannot import name
  'BeforeToolCallEvent' from 'strands.hooks'`.
- **1.13.0** is where `interrupt` enters the event's method resolution order
  (via `HookEvent` and `_Interruptible`), and where `cancel_tool` becomes a
  field the SDK's tool executor translates into a tool-result error. Those are
  the two members the gate actually calls.

The floor is therefore 1.13.0, not 1.10.0. The band between them is the one
worth naming: the class is exported, the import succeeds, nothing refuses at
resolve time or at start-up, and `event.interrupt(...)` raises `AttributeError`
the first time a tool call would move a real robot. A floor recording only where
the *name* arrived would admit exactly that band, which is a worse outcome than
refusing the install.

The bound is declared in `project.dependencies`, in the `[ollama]` extra that
re-declares it, and in `uv.lock`'s transcription of both. The resolved
`strands-agents` in the lock is unchanged, so no dependency version moves.
`tests/test_strands_agents_floor_ships_the_imported_api.py` owns the measurement
and now also grades the members the gate calls, so a release that exports the
name without them cannot pass for one that ships the capability.
