# `rogue_11_resume_token_replay`

**AV-RESUME-HMAC — unauthorised resume of a locked fleet.**

## Threat narrative

Once an emergency-stop has fired, the only thing standing between
the attacker and a re-armed robot is the resume HMAC. The defended
posture binds resume to:

* `STRANDS_MESH_OVERRIDE_CODE` known only to the operator
* A per-envelope `proof_nonce` (anti-replay)
* `HMAC(local_code, proof_nonce)` recomputed at the receiver

## What this rogue does

Four variants:

1. **Fail-closed default** — receiver has no `OVERRIDE_CODE` set;
   every resume is rejected, lockout persists.
2. **No proof + no nonce** — envelope shape gate trips.
3. **Wrong HMAC** — attacker guesses a code, proof mismatches.
4. **Missing proof_nonce** — freshness binding bypass attempt.

In every case the post-condition is `_estop_lockout.is_set() == True`.

## Defences in scope

* `core.Mesh._on_safety_resume`:
  - reads `STRANDS_MESH_OVERRIDE_CODE`; absent → fail-closed.
  - requires `proof_nonce` AND `override_proof` strings.
  - constant-time `hmac.compare_digest` against `HMAC(code, nonce)`.
  - F18-A: HMAC input binds the full envelope shape (issuer_id,
    envelope_t, lockout_elapsed_s) so a captured proof for one
    incident cannot be replayed on a second.

## Pass criterion

All four sub-cases leave the lockout engaged.

## Cross-references

* Source: `core.Mesh._on_safety_resume`.
* Review thread: F18-A (HMAC bound to envelope shape),
  F11/F12 series (resume freshness window).
