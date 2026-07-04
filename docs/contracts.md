# Tool result contract

Every tool-shaped return in `strands_robots` (any `@tool` function, any method
the simulation/hardware routers dispatch to an agent) follows one shape. Keeping
to it is what lets an agent chain tools, reason about telemetry, and lets
downstream consumers (mesh, dashboards, evals) parse results with a fixed schema.

## The shape

A tool result is a dict with exactly two author-controlled top-level keys:

- `status`: one of `"success"`, `"error"`, `"degraded"`.
- `content`: a non-empty list of content blocks. Each block carries exactly one
  of `"text"` (human-readable summary), `"json"` (structured payload), or
  `"image"`.

The Strands runtime may additionally inject a `toolUseId`; authors never set any
other top-level key.

```python
return {
    "status": "success",
    "content": [
        {"text": "Rollout completed: 2 episodes, 1.0 success rate."},
        {"json": {"episodes": episodes, "success_rate": 1.0, "steps": 320}},
    ],
}
```

The canonical reference implementation is
[`strands_robots/tools/run_policy.py`](https://github.com/strands-labs/robots/blob/main/strands_robots/tools/run_policy.py),
which returns `content: [{"text": summary_line}, {"json": payload}]` so latency
masking is provable from the payload, not the logs.

## Why extra top-level keys are a bug

It is tempting to return numeric telemetry as extra top-level keys:

```python
# WRONG - frames/errors/hz_actual are dropped by the runtime
return {
    "status": "success",
    "content": [{"text": "Teleoperation completed: 512 frames, 0 errors."}],
    "frames": 512,
    "errors": 0,
    "hz_actual": 49.7,
}
```

The agent runtime serializes only `status` and `content` into the LLM turn.
Anything outside `content` never reaches `agent.messages`, so the agent cannot
reason about it (for example, deciding whether a teleop session is healthy), and
structured consumers never see it. Put telemetry in a `{"json": {...}}` block:

```python
# RIGHT - telemetry rides in a json content block
telemetry = {"frames": 512, "errors": 0, "hz_actual": 49.7}
return {
    "status": "success",
    "content": [
        {"text": "Teleoperation completed: 512 frames, 0 errors, 49.7Hz."},
        {"json": telemetry},
    ],
}
```

## Choosing `status`

- `"success"`: the operation completed as intended.
- `"error"`: the operation failed (no useful result produced).
- `"degraded"`: the operation ran but with partial failures worth surfacing (for
  example, a teleop loop that produced frames but logged some per-frame errors).

## Enforcement

`tests/test_tool_result_contract.py` statically scans every tool-result-shaped
dict literal in the package and fails if any carries an extra top-level key, so
the contract cannot regress silently. A `**spread` inside a tool-result dict is
flagged too: it can inject arbitrary top-level keys the runtime drops, so it is
never valid at the top level (spread the telemetry into a `{"json": {...}}`
content block instead). The `assert_strands_tool_result` helper in
`tests/tool_result_contract.py` validates a live result and can be applied in any
test that exercises a tool method.
