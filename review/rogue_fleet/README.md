# rogue_fleet — strands-robots PR #195 fleet-style pentest kit

14 process-isolated rogue robots, each one weaponising a different attack
vector against a real `strands_robots.mesh.Mesh` victim. Companion to
`pentest_mesh.py` (which covers the same ground in unit-style form within a
single process).

**Why a fleet form**: every rogue is its own pid + tempdir, fired from a
orchestrator that brings up a clean Mesh victim per scenario. Promotes the
harness from "in-process assertions" to "would survive a real two-host run".
Each rogue ships its own README explaining the threat narrative, defence in
scope, and pass criterion — readable end-to-end as a security audit.

## TL;DR

```bash
# from the repo root, with the dev extras installed:
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[mesh,dev]" cryptography
python review/rogue_fleet/run_fleet.py
# expect: 14/14 defences held in ~9 seconds
```

## Layout

```
rogue_fleet/
├── README.md                           <- this file
├── XRAY.md                             <- codebase data-flow + attack-surface map
├── FLEET_RESULTS.md                    <- generated; latest run summary
├── run_fleet.py                        <- orchestrator
├── _lib/                               <- shared helpers (PKI, reporter, zenoh cfg)
├── target/                             <- victim_robot.py + its README
└── rogue_NN_<name>/
    ├── attack.py                       <- the rogue process (subprocess entry-point)
    ├── README.md                       <- threat narrative + defence + pass criterion
    └── metadata.json                   <- av_id, title, victim/rogue env overrides
```

## The 14 rogues

| #  | Rogue                          | What it attacks                          | Layer            |
| -- | ------------------------------ | ---------------------------------------- | ---------------- |
| 01 | no_cert_outsider               | LAN attacker with no PKI material        | mTLS / wire      |
| 02 | rogue_ca_insider               | Attacker with self-signed cert chain     | mTLS / wire      |
| 03 | namespace_hopper               | Cross-fleet routing isolation            | namespace        |
| 04 | jumbo_dos_publisher            | 32 KiB cmd payload (F1 surface)          | low_pass_filter  |
| 05 | safety_replay_attacker         | Captured estop + peer_id permutation     | safety/estop     |
| 06 | safety_envelope_forger         | Missing/stale/forward `t`, missing peer_id | safety/estop   |
| 07 | acl_role_violator              | enabled:false / interfaces:[] / CN-only  | ACL              |
| 08 | audit_log_tamperer             | HMAC tamper / seq gap / unsigned / PSK rotation | audit log |
| 09 | command_payload_fuzzer         | 7 validate_command bypasses              | payload validator|
| 10 | policy_host_pivot              | CIDR allowlist + IPv6 server_address     | payload validator|
| 11 | resume_token_replay            | Bad HMAC, missing nonce, fail-closed     | safety/resume    |
| 12 | response_hijacker              | Forged RPC response (D1)                 | RPC              |
| 13 | safety_rate_flooder            | Per-issuer fairness on novel-t flood     | safety/estop     |
| 14 | iot_bootstrap_mitm             | Rogue CA + env-var bypass scoping        | IoT bootstrap    |

For the trust-boundary breakdown of why each rogue exists, read
[XRAY.md](./XRAY.md).

## How `run_fleet.py` works

For each rogue:

1. **Tempdir**: fresh per-scenario directory (PKI, ACL, audit dir, results).
2. **PKI**: ephemeral CA + leaf certs (victim, operator) via
   `_lib.pki.EphemeralCA` (re-uses the in-tree `tests/mesh/_pki.py`).
3. **Free port**: random localhost port allocated for this scenario.
4. **Victim spawn**: forks `target/victim_robot.py` with the composed env.
   Blocks on the victim's `READY` line (10s timeout).
5. **Rogue spawn**: forks `rogue_NN_<name>/attack.py` with `VICTIM_LISTEN`,
   `CA_CERT`, `OPERATOR_CERT`, `OPERATOR_KEY`, `ROGUE_RESULT_FILE` env vars.
6. **Wait + collect**: rogue runs to completion, writes a `RogueResult`
   (JSON line) to its result file.
7. **Cleanup**: SIGTERM the victim; rmtree the scenario dir (unless
   `--keep-tmp`).
8. **Aggregate**: at the end, write `FLEET_RESULTS.md` with the table.

Exit code 0 ⇔ every defence held.

### Useful flags

* `--rogue rogue_05_safety_replay_attacker` — run one rogue.
* `--filter safety` — run rogues with that substring.
* `--keep-tmp` — leave scenario tempdirs in `/tmp/rogue_<id>_*` (PKI, audit
  log, result.jsonl) so you can inspect what the rogue did.

## Threat model coverage

The fleet covers the same ground as `pentest_mesh.py` plus three
additional vectors that are easier to express in process-isolated form
(rogue 11 — resume HMAC, rogue 13 — per-issuer fairness,
rogue 14 — IoT bootstrap CA pin). See `XRAY.md §9` for the
cross-reference table and `XRAY.md §10` for explicit non-coverage.

## Per-rogue READMEs

Each rogue has its own README documenting:

* The threat narrative (who is the attacker, what do they want).
* Exactly what the attack script does (step by step).
* Which defences are in scope, with line-number references.
* The pass criterion the rogue expects.
* How the defence would fail (regression mode).
* Cross-references back to `pentest_mesh.py` and to the source.

Reading them in order is intended to read as a security audit of the
entire mesh attack surface.

## Adding a new rogue

```bash
mkdir -p rogue_15_my_attack
cat > rogue_15_my_attack/metadata.json <<JSON
{"av_id": "AV-NEW", "title": "...", "needs_victim": true,
 "victim_env": {}, "rogue_env": {}}
JSON
# Copy attack.py from the closest existing rogue, adapt.
# Write a README.md following the template above.
python run_fleet.py --rogue rogue_15_my_attack
```

## How this complements `pentest_mesh.py`

* `pentest_mesh.py` runs every AV in a single process — fast, dense,
  high-coverage — ideal for CI gating.
* `rogue_fleet/` runs each AV in its own pid against a real victim —
  slower (~9s for the full set), but exercises the wire-level integration
  faithfully. Closest to what a multi-host run would look like.

Both are operator-side; neither lives in the PR diff. Findings get
promoted to in-tree pin tests in `tests/mesh/` when they catch a
regression.
