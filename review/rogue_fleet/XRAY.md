# Strands-robots mesh — codebase X-ray & attack-surface map

*Companion to PR #195 (`mesh/zenoh-native-security`). Trace-first read of every
data path that touches the wire, plus where each rogue in this fleet attaches.*

**Layout**: top-down by trust boundary. Each section is one boundary, listing
the code that owns it, the env vars / config inputs that shape it, the threat
model it defends against, and which `rogue_NN_*` script attacks it.

---

## 0. The fleet bus shape (one diagram)

```
                          ┌────────────────────┐
  Operator (CN: operator-1)──│ Zenoh wire bus  │── Robot R (CN: victim-r1)
          │                 │  tls + ACL +    │         │
          │                 │  namespace +    │         │
  Mesh.send/broadcast        │  low_pass +     │  Mesh._on_cmd
  validate_command (caller)  │  downsampling   │  validate_command (callee)
          │                 └────────────────────┘         │
          │                                              │
          │ strands/<robot>/cmd            <- ACL gate    │
          │ strands/<robot>/response/**    <- D1 scope    │
          │ strands/safety/estop           <- replay+rate │
          │ strands/safety/resume          <- HMAC+nonce  │
          │ strands/<robot>/state          <- read-mostly │
          └────────────────────────────────────────────────────────────────┘
```

Everything below is one of those arrows.

---

## 1. Bus access — "can I send any byte at all?"

**Defended in**: `_zenoh_config.tls_block`, `_zenoh_config.link_protocols_block`,
`_zenoh_config.namespace_block`, `_zenoh_config.scouting_block`.

**Threat model**: A LAN attacker can open sockets to the victim's listen port.
Three attempts:

* **No certs** — plain TCP probe.
  Defence: `link_protocols=["tls"]` makes the listener only TLS.
  **Rogue**: `rogue_01_no_cert_outsider`.
* **Self-signed cert chain** — attacker stands up their own CA, mints `CN=operator-1`.
  Defence: `enable_mtls=true` + explicit `root_ca_certificate` + `verify_name_on_connect=true`.
  **Rogue**: `rogue_02_rogue_ca_insider`.
* **Wrong fleet** — same CA, different `namespace`.
  Defence: `namespace` field. Zenoh prepends + strips at routing layer.
  **Rogue**: `rogue_03_namespace_hopper`.

All three are pre-authorisation — no application byte ever flows.

---

## 2. Authorisation — "now that I have a cert, what may I do?"

**Defended in**: `_acl_config.acl_block`, `_acl_config._validate_acl_shape`,
`_acl_config._is_permissive_acl_shape`, `_acl_config._load_acl_file`.

**Threat model**:
* The default ACL is permissive (any CA-signed peer publishes anywhere) — fine
  for dev, dangerous in prod. The mesh **refuses to start** under
  `auth_mode=mtls + permissive_default_acl` unless
  `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` is explicit.
* Operator-supplied ACLs go through a strict shape validator that catches
  silent-degrade patterns Zenoh would otherwise treat as no-ops:
  - `enabled: false` → ValueError at load.
  - `interfaces: []` → ValueError at validate (the empty-list-matches-nothing trap).
  - **F2 happy path**: `interfaces` field can be omitted (becomes Zenoh wildcard).
* `_is_permissive_acl_shape` shape-detects an operator file that *looks*
  like the dev default, so the start-time gate fires regardless of file
  presence (F18-B fix).

**Rogue**: `rogue_07_acl_role_violator` (3 shape variants + permissive detector).

---

## 3. Per-key-expression rate / size caps — "how loud may you be?"

**Defended in**: `_zenoh_config.low_pass_filter_block`,
`_zenoh_config.downsampling_block`.

**Threat model**: An *authorised* peer (got past mTLS + ACL) is still subject
to Zenoh-level rate / size caps. Without these:

* A 32 MiB cmd payload OOMs the receiver's JSON parser.
* A 100 Hz safety/estop stream exhausts audit-log throughput and
  per-issuer fairness budget.

**Caps applied**:
* `STRANDS_MESH_MAX_CMD_BYTES` (default 16 KiB) on `**/cmd` and `**/broadcast`.
* `STRANDS_MESH_MAX_SAFETY_BYTES` (default 4 KiB) on `**/safety/**`.
* `STRANDS_MESH_MAX_CAMERA_BYTES` (default 1 MiB) on camera frames.
* `STRANDS_MESH_CMD_RATE_HZ` (default 20 Hz) and `STRANDS_MESH_SAFETY_RATE_HZ`
  (default 2 Hz) at the `downsampling` layer.

**F1 fix**: pre-fix code enumerated NICs with a hardcoded fallback. Hosts with
names not in the list (`enp0s3`, `wlP1p1s0`, `cni0`, `wg0`) silently bypassed.
Post-fix: no `interfaces` field → Zenoh treats as wildcard.

**Rogue**: `rogue_04_jumbo_dos_publisher` (size cap on cmd),
`rogue_13_safety_rate_flooder` (per-issuer fairness on novel-t flood).

---

## 4. Payload validation — "does this dict make any sense?"

**Defended in**: `strands_robots/mesh/security.py`.

**Threat model**: An authorised peer publishes a syntactically valid envelope
with semantically dangerous content:

* 200 KB instruction string (LLM prompt-stuffing DoS).
* Inference pivot: `policy_host=evil.attacker.com`.
* HF model path traversal: `pretrained_name_or_path=../../etc/passwd`.
* Unallowed HF org: `evil/backdoor` instead of `lerobot/...`.
* 24-hour `execute` (`duration=86400`).
* Raw string command, not a dict (R24-B — `_dispatch` would crash).
* Unknown action verb.
* Half-spec'd policy (`policy_type` without `policy_provider`).

**Defences**:
* `MAX_*` length / duration / timeout constants.
* `_MODEL_PATH_RE` rejects shell metacharacters and traversal.
* `is_safe_policy_host` (loopback default + CIDR/host allowlist via
  `STRANDS_MESH_POLICY_HOST_ALLOW`).
* `_DEFAULT_POLICY_TYPES` + `STRANDS_MESH_POLICY_TYPE_ALLOW` action verb registry.
* `_DEFAULT_POLICY_HOSTS` + `STRANDS_MESH_HF_REPO_ALLOW` HF-org allowlist.
* `is_safe_server_address` IPv6 + port + bracket parsing (F18-A patch).

**Rogue**: `rogue_09_command_payload_fuzzer` (7 variants),
`rogue_10_policy_host_pivot` (12 host-allowlist + parser cases).

---

## 5. Safety topic — "can you halt the fleet?"

**Defended in**: `core.Mesh._on_safety_estop`, `core.Mesh._on_safety_resume`,
`core._evict_replay_cache`.

### 5.1 Estop — the receiver gauntlet

```
Sample arrives -> _on_safety_estop:
  1. JSON shape gate (peer_id is non-empty str, t is float)
  2. Freshness window check: now - freshness_s <= t <= now + skew_s
  3. Replay cache key=t lookup -- _estop_replay_cache: dict[float_t, (issuer_id, mono_ts)]
  4. Per-issuer fairness derived from cache contents (F8-A / F9-A)
  5. Engage _estop_lockout, audit log, broadcast.
```

* **Step 1 attacks**: missing `t`, missing/empty `peer_id`, malformed JSON.
* **Step 2 attacks**: stale `t`, forward-skewed `t` (lock the fleet forever).
* **Step 3 attacks**: pure replay of captured envelope, peer_id permutation
  (cache keys on `t` alone -- F8-A).
* **Step 4 attacks**: novel-t flood from one issuer.

**Rogues**:
* `rogue_05_safety_replay_attacker` -- replay + peer_id permutation.
* `rogue_06_safety_envelope_forger` -- shape + freshness gate (5 variants).
* `rogue_13_safety_rate_flooder` -- per-issuer fairness.

### 5.2 Resume — the HMAC gate

```
Sample arrives -> _on_safety_resume:
  1. Local STRANDS_MESH_OVERRIDE_CODE configured? else fail-closed.
  2. Envelope has proof_nonce + override_proof strings?
  3. F18-A envelope shape (issuer_id, envelope_t, lockout_elapsed_s) bound to MAC.
  4. hmac.compare_digest( HMAC(local_code, bound_input), provided_proof )
  5. Replay cache check on (issuer, t) so the same proof is consumed once.
  6. Clear lockout, audit, broadcast.
```

**Rogue**: `rogue_11_resume_token_replay` (4 variants: fail-closed, missing proof,
wrong HMAC, missing nonce).

---

## 6. RPC response channel — "is this reply really from R?"

**Defended in**: `core.Mesh._on_response`, `core.Mesh.send`, `core.Mesh.broadcast`.

**Threat model**: `cmd` is locked down by ACL (only operators publish), but
`response/**` is more permissive (any peer can legitimately send replies on
their own turn_ids). An ACL-authorised peer who observes someone else's
turn_id (via the bus or a leak) can publish a forged response on it -- the
sender accepts it as if it came from the legit target.

**Defence**: `_expected_responders[turn_id] = target_peer_id` set in `send()`.
On inbound, `responder_id` is compared against the expected target;
mismatch → drop + audit `response_hijack_rejected`. Broadcast turns use
`BROADCAST_RESPONDER` sentinel and accept any responder.

**Rogue**: `rogue_12_response_hijacker` (D1 — 4 sub-cases).

---

## 7. Audit log — "if it broke, can we tell?"

**Defended in**: `audit.log_safety_event`, `audit.verify_audit_integrity`,
`audit._next_seq`, `audit._sign_record`.

**Threat model**: Post-incident, an attacker on the host wants to scrub
traces of their actions. Five attack patterns:

1. Symlink the log to /dev/null — caught by O_NOFOLLOW + bounded-read
   (`audit.py` writer; covered by AV-21 in `pentest_mesh.py`).
2. Tamper a record's payload — HMAC sig breaks. `bad_signature > 0`.
3. Delete a record — per-peer `seq` gap. `sequence_gaps != []`.
4. Append unsigned record — `missing_sig > 0`.
5. Rotate PSK — verifier-with-old-PSK reports `bad_signature`.

The audit log uses fcntl flock on POSIX so multi-process writes don't tear.
Rotation is bounded by `STRANDS_MESH_AUDIT_MAX_BYTES` and `_MAX_FILES`,
deliberately not via `RotatingFileHandler` (it ignores O_NOFOLLOW).

**Rogue**: `rogue_08_audit_log_tamperer` (4 of the 5 patterns; symlink covered
by `pentest_mesh.py::av_21`).

---

## 8. IoT bootstrap (separate trust boundary)

**Defended in**: `iot/provision.py`. **Lives outside** the mesh wire-layer
but shares the cert-pinning discipline.

**Threat model**: At first run, robots bootstrap through AWS IoT and pull
`AmazonRootCA1.pem` over HTTPS. A network-level adversary (DNS hijack,
captive portal, BGP attack, malicious corp proxy) substitutes a rogue CA
at the canonical URL.

**Defences**:
* `_AMAZON_ROOT_CA1_PINS` -- built-in pin set (64-char lowercase hex).
* `STRANDS_MESH_CA_PINS` -- operator extension; entries that aren't valid
  64-char hex are dropped with WARNING (F3 charset gate).
* The **existing-file branch** ALWAYS pins, regardless of
  `STRANDS_MESH_DISABLE_CA_PIN`. The break-glass is download-only.
* O_NOFOLLOW on the on-disk re-read prevents TOCTOU symlink-swap (R22-D).

**Rogue**: `rogue_14_iot_bootstrap_mitm`.

---

## 9. Cross-references with `pentest_mesh.py` (unit-style)

| Layer                 | Rogue (fleet, this dir) | Unit-style harness AV  |
| --------------------- | ----------------------- | ---------------------- |
| mTLS / wire           | rogue_01, rogue_02      | AV-01, AV-02, AV-19, AV-18 |
| Namespace             | rogue_03                | AV-03                  |
| low_pass_filter / F1  | rogue_04                | AV-04, AV-05           |
| ACL shape / F2        | rogue_07                | AV-06, AV-07, AV-08, AV-34 |
| Payload validator     | rogue_09                | AV-09 — AV-15          |
| Policy host / IPv6    | rogue_10                | AV-10, AV-33           |
| Estop replay          | rogue_05                | AV-26, AV-27           |
| Estop freshness       | rogue_06                | AV-28 — AV-31          |
| Resume HMAC           | rogue_11                | (fleet-only)           |
| Response hijack / D1  | rogue_12                | AV-32                  |
| Per-issuer fairness   | rogue_13                | (fleet-only)           |
| Audit log integrity   | rogue_08                | AV-21 — AV-25          |
| IoT CA-pin            | rogue_14                | (fleet-only)           |
| Auth-mode gate        | (orchestrator startup)  | AV-16, AV-17           |

## 10. Things this fleet does NOT cover (and why)

* **Multi-host clock skew**: rogues 06, 11 simulate the freshness arithmetic
  in-process; a real two-host run would need NTP injection.
* **Live ACL fanout role separation**: tracked in upstream Zenoh issue #200;
  see `pentest_mesh.py` notes section. Not a strands-robots bug.
* **JSON5 fuzzing**: replaced by the `json5` PyPI dep in F15-E; a hypothesis
  fuzz suite is the obvious follow-up but is independent of PR #195 scope.
* **Hardware-in-the-loop**: every rogue here is software-only. Real-robot
  attacks (sensor spoofing, actuator jamming) are out of scope for the
  wire-layer audit.
