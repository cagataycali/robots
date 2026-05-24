# `rogue_12_response_hijacker`

**AV-32 / D1 — RPC response hijack between authorised peers.**

## Threat narrative

The `cmd` channel has tight role separation in the production ACL
(only operators publish, only the addressed robot receives). The
`response/**` channel is broader: anyone in the fleet can
legitimately publish responses (think distributed agents emitting
back to a coordinator). That's a great surface for **lateral
mischief**: an authorised peer who observes a turn_id can publish
a forged response and the sender will accept it as if it were the
legitimate target.

This is exactly the kind of post-mTLS insider attack that gets
blamed on "the system" rather than the operator who configured it.

## What this rogue does

In-process simulation: stand up a `Mesh` as the *sender* (operator),
register a pending P2P turn with `_expected_responders[turn] = "R"`,
then call `_on_response` with:

1. **Forged response** — `responder_id="hijacker"`. Must be dropped.
2. **Legit response** — `responder_id="R"`. Must be accepted.
3. **Broadcast turn** — sentinel allows any responder. Accepted.
4. **Unknown turn_id** — silently ignored (no state pollution).

## Defences in scope

* `Mesh._on_response`:
  - registers `_expected_responders[turn] = target` in `send()`.
  - on inbound, compares `responder_id == expected`; mismatch → drop
    + emit `response_hijack_rejected` audit event.
  - `BROADCAST_RESPONDER` sentinel for `broadcast()`.
* No-state-pollution path for unknown turns (the dispatch table
  filter happens before `_responses.setdefault`).

## Pass criterion

All four sub-cases match expected outcome.

## Cross-references

* Unit-style: `pentest_mesh.py::av_32_response_hijack`.
* Source: `core.Mesh._on_response`.
