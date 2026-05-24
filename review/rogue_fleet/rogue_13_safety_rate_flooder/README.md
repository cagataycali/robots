# `rogue_13_safety_rate_flooder`

**AV-SAFETY-RATE — novel-t flood + per-issuer fairness.**

## Threat narrative

The replay cache (`rogue_05`) catches *repeats* of the same `t`.
It does **not** catch a stream of *novel* `t` values. An attacker
authorised on the safety topic could mint a new envelope every
millisecond — each is fresh, each clears the cache check, each
fires lockout. Beyond just being annoying, this exhausts:

* the receiver's audit-log throughput,
* the per-issuer fairness budget (when implemented),
* legitimate operators' ability to be heard amid the flood.

## Defences in scope

* **Transport-level rate cap** (`downsampling` block on
  `safety/**` topics, default `STRANDS_MESH_SAFETY_RATE_HZ=2`)
  — Zenoh interceptor drops over-rate samples before they reach
  Python.
* **Per-issuer fairness** derived from the replay cache contents
  (F8-A / F9-A). When one peer fills more than its share of the
  `_estop_replay_cache` budget, additional novel-t envelopes from
  the same peer are dropped at the receiver, even if they would
  individually pass the cache check.

## What this rogue does

In-process: shrinks `STRANDS_MESH_RESUME_REPLAY_CACHE_MAX` to 8 to
bring the test into a tight budget. Then:

1. One issuer (`FLOODER`) sends `4 × cache_max` novel-t envelopes
   in tight succession.
2. A second issuer (`good-operator`) sends ONE legitimate envelope
   afterwards.

Fairness check passes when:
* The flooder did NOT engage on every iteration (some were dropped
  by the per-issuer guard).
* The good operator's single envelope still fires lockout.

## Pass criterion

Both conditions hold simultaneously.

## Cross-references

* Source: `core.Mesh._on_safety_estop`, `core._evict_replay_cache`,
  `_zenoh_config.downsampling_block`.
* Review history: F8-A (cache-as-source-of-truth for fairness),
  F9-A (removed redundant `_estop_replay_per_issuer` dict).
