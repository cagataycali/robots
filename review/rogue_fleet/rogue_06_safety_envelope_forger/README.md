# `rogue_06_safety_envelope_forger`

**AV-28 / AV-29 / AV-30 / AV-31 — envelope freshness + shape gates.**

## Threat narrative

Attacker has the basic JSON shape of a `safety/estop` envelope but
tries to game the freshness or identity fields:

* **No `t` field** — hopes the receiver defaults to "now" and
  accepts the envelope.
* **`t` from days ago** — hopes there is no upper bound on age.
* **`t` from days in the future** — hopes the receiver clamps but
  still fires (locking the robot indefinitely).
* **No `peer_id`** — hopes the receiver defaults the issuer (which
  would let one forged envelope drain the per-issuer quota).
* **Empty `peer_id`** — same goal but evades the dict-key check.

All five must be rejected for the freshness gate to be intact.

## Defences in scope

* `core.Mesh._on_safety_estop` -- the freshness/shape preamble:
  - reads `STRANDS_MESH_RESUME_FRESHNESS_S` (default ~30s) and
    `STRANDS_MESH_RESUME_FORWARD_SKEW_S` (default ~5s)
  - rejects envelopes outside `[now - freshness, now + skew]`
  - rejects envelopes with empty / missing `peer_id`
  - rejects envelopes with non-numeric `t`
* These checks happen BEFORE the replay cache is consulted, so a
  forgery never poisons the cache.

## What this rogue does

For each variant, spawns a fresh `Mesh` (so cache state is clean),
delivers a forged sample to `_on_safety_estop`, and asserts
`_estop_lockout.is_set() == False`.

## Pass criterion

All five envelope variants leave `_estop_lockout` cleared.

## Cross-references

* Unit-style: `pentest_mesh.py::av_28_estop_no_timestamp`,
  `av_29_estop_stale_t`, `av_30_estop_forward_skew`,
  `av_31_estop_no_peer_id`.
* Source: `core.Mesh._on_safety_estop` (lines that read `t`,
  `peer_id`, validate freshness).
