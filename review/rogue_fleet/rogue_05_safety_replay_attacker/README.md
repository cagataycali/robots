# `rogue_05_safety_replay_attacker`

**AV-26 / AV-27 — captured-envelope replay + peer_id permutation.**

## Threat narrative

`safety/estop` is the highest-privilege topic in the fleet bus: a
single valid envelope locks down every robot until an operator runs
the resume protocol. That makes it the most attractive replay target
on the entire wire.

The attacker has captured one legitimate envelope (off the bus during
a previous incident, from a leaked audit log, from a decommissioned
peer's disk, etc.). Two attacks they would try:

1. **Pure replay** — reissue the captured envelope verbatim.
2. **`peer_id` permutation** — modify only the payload `peer_id`
   field, hoping the receiver dedupes by `(peer_id, t)` and not by
   `t` alone.

## Defences in scope

* `core.Mesh._estop_replay_cache: dict[float_t, (issuer_id, mono_ts)]`
  — keyed by **timestamp `t` only**. Permuting payload `peer_id`
  does not change `t`, so the second arrival hits the cache.
* F8-A: per-issuer fairness counts derived from cache contents.
* F9-A: removed the redundant `_estop_replay_per_issuer` dict;
  fairness derived from the canonical cache.
* Resume freshness window (`STRANDS_MESH_RESUME_FRESHNESS_S`,
  default 30 s) bounds how long a captured envelope is replayable
  even if the cache is cold.

## What this rogue does

Spins up a real `Mesh` in-process, then calls `_on_safety_estop`
three times with hand-forged Zenoh samples:

| Step | Envelope                                  | Expected outcome |
| ---- | ----------------------------------------- | ---------------- |
| 1    | `{"peer_id":"operator-1","t":T0}`         | lockout engaged  |
| 2    | same `t`, same payload                    | dropped by cache |
| 3    | same `t`, **`peer_id` permuted to ...-2`**| dropped by cache |
| 4    | new `t`=T0+0.1, fresh envelope            | lockout engaged  |

If any step deviates, the defence is bypassed.

## Pass criterion

All of:
* Step 1 sets `_estop_lockout`.
* Step 2 leaves `_estop_lockout` cleared.
* Step 3 leaves `_estop_lockout` cleared (this is the AV-27 surface).
* Step 4 sets `_estop_lockout` (legitimate path still works).

## Failure modes

* If `_estop_replay_cache` keys on `(peer_id, t)`, step 3 bypasses.
* If the cache is cleared on lockout-clear (legit resume), step 2
  bypasses.
* If freshness window is unbounded, captured envelopes are
  forever-valid.

## Cross-references

* Unit-style: `pentest_mesh.py::av_26_estop_replay`,
  `pentest_mesh.py::av_27_estop_peer_id_permutation`.
* Source: `core.Mesh._on_safety_estop`,
  `core._evict_replay_cache`.
* Review history: F8-A (cache-as-source-of-truth),
  F9-A (per-issuer-fairness derived).
