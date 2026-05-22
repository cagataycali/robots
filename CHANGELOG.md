# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here. Follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased - #194 (mesh security hardening, Zenoh-native rewrite)

### Final refactor — drop the application-layer envelope, lean on Zenoh built-ins

Earlier rounds of this PR hand-rolled an HMAC + nonce + per-peer-key
envelope on top of Zenoh. The final round replaces that entire stack
with Zenoh's built-in primitives, which provide stronger guarantees
with less code:

* **Authentication** — Zenoh ``transport/link/tls`` (mutual TLS) binds
  peer identity at the link layer. Cert Common Name encodes role
  (``robot-*``, ``op-*``, ``audit-*``). No application crypto.
* **Authorisation** — Zenoh ``access_control`` enforces per-key-
  expression ACLs on cert CN. Default-deny.
* **Discovery** — multicast scouting off by default; gossip-only with
  explicit endpoints. Closes the LAN-attacker-enrollment surface
  the old PSK envelope did not.
* **Fleet isolation** — Zenoh ``namespace`` prepends a fleet prefix
  at the routing layer so two fleets cannot collide.
* **Rate / size caps** — Zenoh ``downsampling`` (per-key freq) and
  ``low_pass_filter`` (per-message bytes) at the transport,
  pre-deserialise. Floods cost the receiver nothing.

What survives the rewrite (payload semantics, not wire auth):

* ``mesh.security.validate_command`` and ``is_safe_*`` allowlists.
* ``mesh.audit`` HMAC-signed audit log.
* ``Mesh._on_response`` ``responder_id`` binding (point-to-point
  scope check between authorised peers).
* ``STRANDS_MESH_OVERRIDE_CODE`` second factor for fleet-wide resume.
* ``mesh.transport.bridge_transport`` cross-transport dedup.

Lines deleted (net):

* ``strands_robots/mesh/identity.py`` — entire 505-line module.
* ``strands_robots/mesh/security.py`` — shrank from 1176 to 467
  lines. Dropped: ``sign_envelope`` / ``verify_envelope``,
  ``_NONCE_CACHE``, ``TokenBucket``, ``pin_peer_identity`` /
  ``drop_peer_identity`` / ``known_peer_identities``,
  ``_PROCESS_STATE``, ``_canonical_bytes``, ``_hmac_hex``,
  ``RateLimitError``, ``AuthenticationError``,
  ``psk_configured`` / ``auth_required`` / ``identity_required``.
* ``strands_robots/mesh/core.py`` — envelope sign / verify gone from
  every ``_on_*`` handler, ``_put_signed`` is now a thin
  ``put(key, payload)`` wrapper, ``_on_safety_estop`` /
  ``_on_safety_resume`` no longer gate on PSK.
* Tests — ``tests/mesh/test_peer_identity.py`` and the envelope-only
  parts of ``test_security.py`` / ``test_security_regressions.py``
  deleted; replaced by ``test_validate_command.py`` (44 tests),
  ``test_zenoh_config.py`` (28 tests), ``test_acl_config.py`` (13
  tests), and ``test_session_config.py`` (18 integration tests).
  ``test_pentest_findings.py`` rewritten to keep the D1 / A2 / E1 /
  F1-3 / G1-2 regressions that survive the wire-layer change.

Lines added:

* ``strands_robots/mesh/_zenoh_config.py`` (~360 lines, pure config builders).
* ``strands_robots/mesh/_acl_config.py`` (~310 lines, ACL builder + JSON5-lite loader).
* ``examples/mesh_acl_example.json5`` (canonical reference ACL).

Net: roughly 1700 lines deleted, 670 lines added; 1030-line reduction
in the mesh tree, all defensive guarantees preserved or tightened.

### Removed env vars (breaking)

| Removed | Replacement |
|---|---|
| ``STRANDS_MESH_PSK`` | ``STRANDS_MESH_TLS_*`` (mTLS cert paths) |
| ``STRANDS_MESH_REQUIRE_AUTH`` | ``STRANDS_MESH_AUTH_MODE=mtls`` (default) |
| ``STRANDS_MESH_REQUIRE_PEER_IDENTITY`` | mTLS handshake is identity proof |
| ``STRANDS_MESH_PEER_IDENTITY`` | n/a (no per-peer HMAC scheme) |
| ``STRANDS_MESH_PEER_KEY`` | mTLS private key |
| ``STRANDS_MESH_PEER_KEY_FILE`` | ``STRANDS_MESH_TLS_KEY`` |
| ``STRANDS_MESH_PEER_KEY_DIR`` | n/a |
| ``STRANDS_MESH_REPLAY_WINDOW`` | TLS per-link sequence numbers |
| ``STRANDS_MESH_PEER_RATE`` | ``STRANDS_MESH_CMD_RATE_HZ`` (transport-layer ``downsampling``) |

### New env vars

| Var | Default | Purpose |
|---|---|---|
| ``STRANDS_MESH_AUTH_MODE`` | ``mtls`` | ``mtls`` (prod) or ``none`` (dev only) |
| ``STRANDS_MESH_TLS_CA`` | unset | CA bundle for peer-cert verification |
| ``STRANDS_MESH_TLS_CERT`` | unset | this peer's cert (PEM) |
| ``STRANDS_MESH_TLS_KEY`` | unset | this peer's private key (PEM, 0o600) |
| ``STRANDS_MESH_ACL_FILE`` | unset | JSON5 ACL override; defaults to built-in |
| ``STRANDS_MESH_NAMESPACE`` | ``strands_robots`` | fleet prefix |
| ``STRANDS_MESH_MULTICAST`` | ``false`` | gossip-only by default |
| ``STRANDS_MESH_MAX_SESSIONS`` | ``256`` | DoS bound |
| ``STRANDS_MESH_MAX_CMD_BYTES`` | ``16384`` | jumbo-frame DoS bound |
| ``STRANDS_MESH_MAX_CAMERA_BYTES`` | ``1048576`` | camera frame cap |
| ``STRANDS_MESH_CMD_RATE_HZ`` | ``20.0`` | per-key cmd rate cap |

### Kept env vars

``STRANDS_MESH_AUDIT_PSK`` (audit log signing — independent of wire
auth), ``STRANDS_MESH_AUDIT_DIR``, ``STRANDS_MESH_AUDIT_MAX_BYTES``,
``STRANDS_MESH_AUDIT_MAX_FILES``, ``STRANDS_MESH_OVERRIDE_CODE``
(operator second factor for fleet resume), ``STRANDS_MESH_DEDUP_TTL``,
``STRANDS_MESH_BRIDGE_TOPICS``, ``STRANDS_MESH_BRIDGE_TOPICS_PREFIX``,
``STRANDS_MESH_HF_REPO_ALLOW``, ``STRANDS_MESH_POLICY_HOST_ALLOW``,
``STRANDS_MESH_POLICY_TYPE_ALLOW``, ``STRANDS_MESH_CAMERA_*``,
``STRANDS_MESH_CA_PINS``, ``STRANDS_MESH_DISABLE_CA_PIN``.

### Migration

For an existing deployment running under the old PSK envelope:

1. Generate a CA + per-thing cert chain (or reuse the AWS IoT certs
   already in ``~/.strands_robots/iot/`` — they meet the same shape).
2. Set ``STRANDS_MESH_TLS_CA`` / ``CERT`` / ``KEY`` on every peer.
3. Update peer-id naming so it matches a CN glob in the ACL: robots
   become ``robot-<id>``, operators ``op-<id>``.
4. Drop all ``STRANDS_MESH_PSK`` / ``STRANDS_MESH_PEER_KEY*`` vars.
5. Drop ``~/.strands_robots/mesh/`` (legacy per-peer key store).

A dev / lab environment without PKI can run ``STRANDS_MESH_AUTH_MODE=none``
to keep the old plain-TCP behaviour. The mesh logs a WARNING.

### Earlier rounds (historical, archived below)

The pre-rewrite rounds are kept for forensic context. Code references
in those entries (``_NONCE_CACHE``, ``identity.py``, ``sign_envelope``,
``STRANDS_MESH_PSK``, etc.) point at code that no longer exists.

## Earlier round entries — pre-rewrite (kept for context)

### Round 8 - final-pass review feedback (yinsong1986)

### Round 8 - final-pass review feedback (yinsong1986)

* **R8-1 (HIGH)**: receiver-side ``_on_safety_estop`` and
  ``_on_safety_resume`` now write audit records via
  ``publish_safety_event``. Pre-fix the receivers only logged at
  CRITICAL/WARNING - ``verify_audit_integrity`` walkers couldn't see
  which peers actually engaged their lockout in response to a fleet-
  wide estop, breaking the per-record HMAC + seq forensic story for
  the most operationally important event. Now both ends of the lockout
  window write ``remote_estop_engaged`` / ``remote_resume_applied``
  audit entries with the issuer peer_id and timestamp.

* **R8-2 (HIGH)**: legacy bare-dict payloads (no ``v`` / ``payload``
  envelope keys) previously bypassed BOTH the freshness window and
  the nonce-cache replay check in permissive mode. An attacker who
  captured any legacy cmd could replay it indefinitely. Fixed by
  synthesizing a content-fingerprint nonce
  (``"L:" + sha256(canonical_bytes)``) BEFORE the early-return so
  identical content within the replay window is rejected by
  ``_NONCE_CACHE``. Distinct legacy contents still pass; freshness
  is still skipped (no ts to compare) - that's residual risk for
  permissive-mode operators, documented inline.

* **R8-3 (LOW cleanup)**: removed the raise-then-immediately-catch
  ``AuthorizationError`` pattern from ``_on_response``. The R4-8
  scaffolding never gained a real consumer; ``logger.debug`` plus the
  structured ``response_hijack_rejected`` audit event are the channels
  forensic readers actually consume. ``AuthorizationError`` class
  deleted entirely from ``mesh/security.py`` per AGENTS.md > Key
  Conventions #10 ("No dead code"). The R4-8 regression test now
  pins the deletion (``test_r4_8_supersedes_authorization_error_deleted_per_r8_3``).

* **R8-4 (BUG)**: audit-log rotation cascade off-by-one. With
  ``max_files=N`` the loop iterated ``[N-1 .. 1]`` and the predicate
  ``n + 1 > max_files - 1`` always discarded the file at ``.{N-1}``
  instead of rolling it to ``.{N}`` - rotated suffixes only ever
  reached ``.{N-1}``. Operators setting ``MAX_FILES=5`` got 4 rotated
  copies, not 5. Fix: walk ``[N .. 1]``, predicate ``n + 1 > max_files``.
  Pinned by ``test_r8_4_rotation_reaches_max_files`` which floods
  enough events to force ``.5`` to exist on disk.

* **R8-5 (HIGH - description vs implementation drift)**:
  ``STRANDS_MESH_OVERRIDE_CODE`` now genuinely protects the fleet-wide
  resume, not just the local issuer's gate. Pre-fix any peer holding
  the PSK could broadcast a ``safety/resume`` and clear every other
  peer's lockout - the override code only ran in
  ``Mesh._resume_lockout`` on the issuing peer, never on the wire.
  Now the issuer binds ``HMAC(override_code, proof_nonce)`` into the
  resume envelope; ``_on_safety_resume`` re-verifies with its OWN
  ``STRANDS_MESH_OVERRIDE_CODE`` (constant-time-compared). Receivers
  without an override code configured FAIL CLOSED and refuse remote
  resumes - operators must distribute the override code to every peer
  for fleet-wide remote resume to work. That asymmetry is intentional
  and documented in ``_on_safety_resume``'s docstring.

* **R8-6 (MED - defence-in-depth)**: ``Mesh.send`` and ``Mesh.broadcast``
  now run client-side ``validate_command`` before publishing.
  Programmatic callers (third-party integrations, tests, scripts that
  import ``Mesh`` directly) previously skipped this gate; only the
  ``robot_mesh`` LLM tool path validated client-side. The receiver
  still validates server-side, so this is defence-in-depth, but the
  PR description and README claimed both sides - now they do.
  ``Mesh.send`` returns the structured ``{"status": "error", "error":
  "validation: ..."}`` shape; ``Mesh.broadcast`` returns ``[]`` on
  rejection (preserves the list-of-responses contract). Crucially,
  the bad cmd never reaches ``_put_signed`` so it never hits the wire.

* **R8-7 (MED - safety ordering)**: for ``broadcast``, the
  ``robot_mesh`` tool now parses + validates the command JSON BEFORE
  raising the HITL interrupt. Pre-fix the operator could approve a
  command that the validator then rejected, burning an audit
  ``operator approved`` record AND a rate-limit slot for an action
  that never ran. The interrupt's ``reason`` dict now surfaces the
  validated command (post-strip-and-lower) so operators approve the
  post-validation form, not raw LLM output. ``emergency_stop`` has
  no command body so its ordering is unchanged.

* **R8-10 (CodeQL #229)**: deleted the legacy ``_AMAZON_ROOT_CA1_SHA256``
  alias. After R7-3 it was only assigned (``= _AMAZON_ROOT_CA1_PINS[0]``)
  and CodeQL's dataflow analyser flagged it as unused. Internal code
  references the tuple directly; error messages format the full pin
  set via ``_resolve_ca_pins`` so operators see every accepted pin
  (not just the canonical first one).

* **R8-polish (LOW)**: three small unresolved findings from R4/R5/R7
  pinned with regression tests:
    - R5-defensive: belt-and-suspenders BROADCAST_RESPONDER guard at
      the ``_expected_responders[turn] = target`` assignment site, in
      addition to the public ``Mesh.send`` entry-point guard. Future
      refactors that bypass the public guard (e.g. an internal helper)
      cannot reopen the response-hijack surface.
    - R7-policy-lc: locked the ``policy_provider`` lowercase contract
      with a regression test (``test_r7_policy_provider_lowercase_contract``).
      ``validate_command`` strips + lowercases; dispatch can rely on
      the canonical key.
    - R5-flood: pinned the permissive-mode nonce-cache flood eviction
      behaviour. A flood of ~1.5x cap unique nonces stays bounded at
      the cap (10 000); GC drops the oldest 20%. Documents the
      acknowledged O(n) GC cost as a residual permissive-mode risk.

Tests: 14 + 3 polish R8 regression tests in test_security_regressions.py.
The R4-8 test was rewritten (``test_r4_8_supersedes_authorization_error_deleted_per_r8_3``)
to pin the deletion. The pre-existing
``test_safety_estop_topic_engages_remote_lockout`` was updated to
include ``override_proof`` (R8-5) and a configured
``STRANDS_MESH_OVERRIDE_CODE`` on the receiver. Two
``test_iot_ca_pin`` tests stayed valid - they patch
``_download_with_per_socket_timeout`` directly.

Stats: 697 mesh tests passing (+17 from R7 baseline of 680, includes 3 R8 polish tests), ruff
clean, mypy clean.

### Round 7 - additional review feedback (yinsong1986)

* **R7-1 (HIGH)**: ``audit.py:_ensure_paths`` had two compounding bugs.
  The convoluted ``try/except OSError`` + re-check pattern could
  silently swallow the symlink rejection on a TOCTOU race, and
  ``Path.touch()`` follows symlinks regardless. Fix: drop the
  try/except wrapper (``Path.is_symlink()`` does not raise on missing
  files), and replace ``Path.touch()`` with
  ``os.open(path, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW, 0o600)`` so the
  create itself refuses to follow a symlink. On Windows where
  ``O_NOFOLLOW`` is ``0`` the static check remains the only line of
  defence (residual risk documented in the module docstring).

* **R7-2 (MED)**: ``provision.py:_ensure_ca`` mutated
  ``socket.setdefaulttimeout`` for the duration of the urlopen — a
  process-global side effect that any concurrent boto3 / Zenoh /
  requests thread observed during the CA window. Fix: new
  ``_download_with_per_socket_timeout`` helper installs a one-shot
  ``urllib.request.HTTPSHandler`` whose ``https_open`` builds
  ``HTTPSConnection`` instances with the timeout baked in. Per-socket
  only; no process-global mutation. The R4-4 invariant (per-recv
  deadline) is preserved with a stricter implementation.

* **R7-3 (operational HIGH)**: hardcoded ``_AMAZON_ROOT_CA1_SHA256``
  was a flag-day time bomb. AWS root rotation would hard-fail every
  deployment with no recovery path short of a code change. Fix:
  promoted to ``_AMAZON_ROOT_CA1_PINS: tuple[str, ...]`` plus a new
  ``STRANDS_MESH_CA_PINS`` env var (comma-separated 64-char lowercase
  hex; invalid entries logged and skipped). Operators stage a future-
  rotation pin via the env var ahead of a code-level rotation; the
  built-in tuple is always included. New ``_resolve_ca_pins`` helper
  centralises the pin-set composition; ``_hash_matches_pin`` /
  ``_verify_ca_bytes`` / ``verify_ca_pin`` all consult it. Backwards-
  compat: ``_AMAZON_ROOT_CA1_SHA256`` remains as the canonical
  (first) pin so existing references keep working.

* **R7-4 (HIGH)**: explicit-``None`` hole in five ``validate_command``
  per-field gates. ``cmd.get(k, default)`` returns ``None`` when the
  key is present with ``None`` value, ``if value`` short-circuited
  every gate, and the explicit-None survived in ``out = dict(cmd)``
  to be forwarded into the executor. Fix: distinguish key-absent
  (apply default) from key-present (must be a non-empty string in
  the allowlist). Pattern applied to ``policy_provider``,
  ``policy_type``, ``model_path``, ``pretrained_name_or_path``, and
  ``server_address``. ``policy_provider``'s back-compat default
  ``"mock"`` is preserved on the absent-key path.

* **R7-5 (LOW)**: ``robot_mesh.py:_audit_tool_action`` had bare
  ``except Exception: pass`` with no log line. Fix: copy the
  ``core.py:_on_cmd`` pattern -- catch with breadcrumb at DEBUG so
  operators investigating "why don't I see my LLM tool actions in
  the audit log?" find a trace. Wide catch is intentional (audit
  failures must NEVER propagate up into the safety code path) and
  documented inline so AGENTS.md > "Exception Clauses Must Be Narrow"
  is not violated implicitly.

New env var: ``STRANDS_MESH_CA_PINS`` (R7-3). Documented in README.

Tests: 13 new R7 regression tests in test_security_regressions.py.
The R4-4 test was updated to reflect R7-2's stricter implementation
(no setdefaulttimeout call inside _ensure_ca).

Stats: 680 mesh tests passing (+13 from R6 baseline of 667), ruff
clean, mypy clean.

### Round 6 - scope-creep cleanup (yinsong1986)

* **Scope creep dropped**: ``.gitignore`` row for
``system_prompt.prompt`` removed (local tooling artifact); the file
remains untracked locally. ``PENTEST_FINDINGS.md`` removed entirely
from this PR (-423 lines). Cycle-by-cycle pentest evidence is
preserved in commit messages, CHANGELOG round-by-round entries, and
inline docstrings on every test in
``tests/mesh/test_pentest_findings.py``.

### Round 5 - senior-principal pass (yinsong1986)

* **R5-1 (HIGH)**: ``_exec_cmd`` receive-side ``turn_id`` fallback was
  ``uuid.uuid4().hex[:8]`` (32-bit). Promoted to full 128-bit hex to
  match the outbound D1 hardening. Pre-fix, an inbound command without
  ``turn_id`` produced a birthday-colliding response topic that an
  observer could predict.

* **R5-2 (MED)**: promoted ``Mesh._put_signed`` to a public
  ``Mesh.publish`` alias for cross-module callers (currently
  ``camera_offload``). AGENTS.md > Public API Hygiene forbids
  referencing ``_method`` names from other modules. ``_put_signed``
  remains the canonical name for in-class and intra-module use;
  ``publish`` delegates.

* **R5-3 (HIGH)**: cross-process safety on the seq sidecar via
  ``fcntl.flock``. Two processes that share an audit dir cannot roll
  the counter back any more — ``_next_seq`` re-reads the sidecar
  inside the inter-process flock, merges any peer-process increments
  into the in-memory cache, increments, persists, releases. POSIX
  only; Windows falls back to in-process locking with the limitation
  documented in the module docstring.

* **R5-4 (HIGH)**: ``_resume_lockout`` collapsed to a single generic
  wire-response shape (``{"status": "ok"}`` on success,
  ``{"status": "error", "error": "resume rejected"}`` on every
  failure). Pre-fix the four distinct response shapes leaked oracles
  for: lockout engaged or not, ``STRANDS_MESH_OVERRIDE_CODE``
  configured or not, and the lockout duration. Structured detail is
  preserved in the local ``publish_safety_event`` audit record where
  forensics can use it.

* **R5-5 (LOW)**: ``_peer_rate_config`` exception clause narrowed from
  bare ``except Exception`` to ``except (ValueError, IndexError)``.
  AGENTS.md > Exception Clauses Must Be Narrow. Real bugs in the
  clamping logic now surface instead of being silently masked.

### Structural cleanup pass

* **No duplicate top-level imports** across the seven hardened modules.
* **Consolidated** the two consecutive ``from strands_robots.mesh.session
  import (...)`` blocks in ``core.py``.
* **Hoisted** the four lazy ``from strands_robots.mesh.audit import
  log_safety_event`` calls inside ``core.py`` (``_on_cmd``, ``_exec_cmd``,
  ``_on_response``) to a single top-level import — ``audit.py`` has no
  dependency on ``core.py`` so the lazy form was unnecessary.
* **Hoisted** the ``import socket as _socket`` lazy import inside
  ``_ensure_ca`` to the top of ``provision.py``.
* **Verified zero emojis** in source code across the seven hardened
  modules.
* **Verified no superfluous one-line comments** that restate the
  obvious — the section dividers in ``robot_mesh.py`` are intentional
  navigation aids and were kept.

### Round 1 / 2 / 3 / 4 - see existing entries below.

### Round 4 — additional review feedback (yinsong1986)

* **R4-1 (HIGH)**: ``policy_provider`` now gated through
  ``is_safe_policy_provider`` in ``validate_command``. Default ``"mock"``
  matches the receive-side dispatcher's default so back-compat callers
  keep working. Shares the allowlist with ``policy_type`` (extend via
  ``STRANDS_MESH_POLICY_TYPE_ALLOW``). Threat-vector #3 was reachable
  via ``policy_provider`` even after R3-5 added gates for the four
  ``policy_type`` / ``model_path`` / ``pretrained_name_or_path`` /
  ``server_address`` fields — Yin caught that ``policy_provider`` is
  the actual registry key the executor uses.

* **R4-2 (HIGH)**: ``_sign_record`` snapshots PSK presence on the first
  call and refuses to write subsequent unsigned records with a new
  ``AuditPSKDegradedError``. ``verify_audit_integrity`` now reports
  ``ok=False`` when ``psk_present and missing_sig > 0``. An attacker
  who briefly clears the env mid-run to write unsigned forgeries can
  no longer hide them behind ``ok=True``.

* **R4-3 (MED)**: ``_persist_seq_counters`` fsyncs the temp fd before
  ``os.replace`` and the parent directory afterwards (POSIX). After a
  power loss, the audit log can no longer have records ahead of the
  sidecar — duplicate seq values on restart are eliminated.

* **R4-4 (MED)**: ``_ensure_ca`` wraps ``urlopen`` with
  ``socket.setdefaulttimeout(15.0)`` so each ``recv()`` observes the
  timeout — previously only connect+handshake were bounded and a
  slow-loris responder could dribble bytes for arbitrary wall-clock.

* **R4-5 (LOW)**: ``Mesh.send`` rejects ``BROADCAST_RESPONDER`` and any
  NUL-containing target up front. Defence-in-depth: a future refactor
  that loosens the peer_id regex cannot reopen the response-hijack
  surface that D1 closed.

* **R4-8 (CLEANUP)**: ``RateLimitError`` and ``AuthorizationError`` are
  no longer dead code. New ``enforce_peer_rate_limit(sender_id)`` is
  the structured form of ``consume_peer_token`` (raises on starvation).
  ``Mesh._on_response`` raises ``AuthorizationError`` on the response-
  hijack reject path AND emits a typed
  ``response_hijack_rejected`` audit event so forensic readers see the
  rejection in a structured field instead of a free-text log line.

* **R4-6 / R4-7**: residual-risk documentation. The permissive-mode
  replay-cache fillability and the audit-write amplification under
  the configured peer-rate ceiling are documented in the module
  docstrings as accepted limitations with mitigation paths.

* **R4-10** (superseded by R6): ``PENTEST_FINDINGS.md`` was removed
  from this PR per scope-creep feedback (Yin, R6). The cycle-by-cycle
  pentest evidence lives in commit messages and in this CHANGELOG; the
  regression tests in ``tests/mesh/test_pentest_findings.py`` carry
  the threat-model docstrings inline.

* **CodeQL #226 / #227 / #228**: explanatory comments added to the
  three previously-bare except-pass blocks (``audit.py:316``,
  ``audit.py:505``, the ``_put_signed`` ``...`` stub in
  ``sensors.py:83`` is now a ``raise NotImplementedError(...)``).

### Round 1 / 2 / 3 — see existing entries below.

### Round 3 — additional review feedback (yinsong1986)

* **Sanitised dispatch-error wire output** (R3-1): the catch-all in
  `Mesh._exec_cmd` no longer leaks `str(exc)` onto the response topic.
  Internal exception detail (filesystem paths, attribute names,
  third-party traces) is logged locally with `exc_info=True` for
  operator debugging; the wire emits only the static string
  `"dispatch error"`. Structured `ValidationError` / `LockoutError`
  paths remain the preferred channel for the rejections clients need
  to distinguish.

* **Released `_PEER_RATE_LOCK` before `bucket.consume()`** (R3-2): the
  per-sender registry lock no longer covers the `bucket.consume()`
  call. TokenBucket has its own internal lock so per-sender
  consumption no longer single-threads through the global registry —
  high command volume across many peers can now actually scale.

* **Declined HITL approvals do NOT consume rate-limit slots** (R3-3):
  `_rate_limit_check` was split into check-and-record. A declined
  operator approval skips the record. Without this, three nuisance
  LLM prompts an operator declines within a minute would lock the
  agent out of issuing a real `emergency_stop` (capped at 3/min).
  Approved actions and non-interrupt actions still consume slots
  unconditionally.

* **CA pin always raw-checked on the on-disk re-use path** (R3-4):
  `STRANDS_MESH_DISABLE_CA_PIN` no longer applies to existing files.
  The break-glass exists for the *download* path (re-encoding
  proxies); silently re-using a rogue CA from a prior compromised
  provisioning run is strictly worse than re-fetching every time.
  Operators who need to refresh a re-encoded cert must delete the
  file to force a fresh download. Documented inline.

* **Extended `validate_command` to gate `model_path`,
  `pretrained_name_or_path`, `policy_type`, and `server_address`**
  (R3-5): the receiver-side `_dispatch` was forwarding these fields
  to `_execute_task_sync`/`start_task` with no validation, so an
  authenticated peer (or a leaked PSK) could pin robots at attacker-
  controlled HF repos, filesystem paths, or remote inference servers
  — bypassing the spirit of the `policy_host` allowlist. New helpers:
  `is_safe_model_path` (charset + traversal check, optional HF org
  allowlist), `is_safe_policy_type` (enum allowlist), and
  `is_safe_server_address` (host portion routed through the existing
  policy-host allowlist).

* **Refactored `_PSK_WARNED` and `_SEQ_LOADED` onto module-level state
  classes** (R3-6, CodeQL #219, #222, #223): the bare module-level
  scalars repeatedly tripped CodeQL's "unused global variable" rule
  even after the helper-hoist refactor. Both flags now live on
  `_PROCESS_STATE.psk_warned` and `_AUDIT_STATE.seq_loaded`
  respectively. Static analysers see a normal attribute read+write
  on a single object instead of a `global` declaration on a scalar.

* **Documented the chmod best-effort `except OSError: pass`** in
  `_persist_seq_counters` (R3-7, CodeQL #225). chmod failure on
  filesystems that don't honour POSIX permissions (FAT32, NFS without
  uid map, restricted-mount volumes) is silently ignored — having a
  working audit log without `0o600` is preferable to crashing safety
  persistence over a chmod failure.

* **Removed the duplicate docstring in `SensorLoopsMixin`** (R3-8,
  CodeQL #224). The class had two consecutive docstrings; the second
  was parsed as a no-effect string-statement. The single canonical
  docstring now sits at the top of the class body, with the
  `_put_signed` stub explicitly noted as runtime-shadowed by the
  `Mesh` host class.

* **New env vars documented** in README.md:
  `STRANDS_MESH_HF_REPO_ALLOW`, `STRANDS_MESH_POLICY_TYPE_ALLOW`.

### Round 1 + 2 — see existing entries below.

### Added: HMAC-signed envelopes for the mesh wire format

Every mesh publish now goes through `Mesh._put_signed`, which wraps the
payload in a versioned envelope: `{v, ts, nonce, payload, sig}`. The
signature is HMAC-SHA256 over a canonical (sort_keys=True) JSON encoding
of the rest of the envelope, keyed by `STRANDS_MESH_PSK`. Receivers call
`mesh.security.verify_envelope` (with a per-peer scope so multiple Mesh
peers in the same process maintain independent replay caches). The
verifier enforces an asymmetric freshness window (60 s past, 5 s
forward), constant-time signature compare, and per-message replay
protection via uuid4 nonces.

Permissive mode (`STRANDS_MESH_PSK` unset) emits envelopes without a
signature and accepts un-enveloped legacy payloads, so existing
zero-config Zenoh-LAN setups keep working. Strict mode is opt-in via
`STRANDS_MESH_PSK` plus optionally `STRANDS_MESH_REQUIRE_AUTH=true`.

### Added: command validation, action allowlist, `policy_host` allowlist

`mesh.security.validate_command` runs both at the receiver
(`Mesh._exec_cmd`) and client-side (in `tools.robot_mesh`'s `tell`,
`send`, `broadcast`). It enforces an action allowlist, instruction
length cap (2 KiB), `duration` ≤ 1 h, `policy_port` in `[1, 65535]`,
`steps` ≤ 10 000, and a `policy_host` allowlist (loopback-only by
default; extend via `STRANDS_MESH_POLICY_HOST_ALLOW` with
hostname or CIDR entries).

### Added: per-sender token-bucket rate limit on `_on_cmd`

`mesh.security.consume_peer_token(sender_id)` is called before
`Mesh._on_cmd` spawns the exec thread. Default 20 cmds/60 s per sender,
configurable via `STRANDS_MESH_PEER_RATE="<count>/<seconds>"` (burst
capped at 1000). Idle senders are GC'd from the registry. Rate-limit
drops are recorded in the audit log.

### Added: emergency-stop persistent lockout (strict mode)

`Mesh.emergency_stop()` engages a thread Event; `_dispatch` raises
`mesh.security.LockoutError` for every action other than `status` or
`resume` until the lockout is cleared. `Mesh.start()` subscribes to
`strands/safety/estop` and `/safety/resume`, so the lockout is fleet-wide
in strict mode. `_resume_lockout` requires `STRANDS_MESH_OVERRIDE_CODE`
(constant-time compared). The lockout is **soft in permissive mode**:
the safety handlers refuse to act on remote events when neither PSK nor
strict-auth is configured, so a LAN attacker cannot weaponise the
lockout for DoS. Documented in README's *Mesh security* section.

### Added: signed audit log + integrity verifier

`mesh.audit.log_safety_event` now writes per-record HMAC signatures
(when `STRANDS_MESH_AUDIT_PSK` is configured) plus per-peer monotonic
sequence numbers. Counters persist to a sidecar file (`mesh_audit.seq.json`
next to the audit log) so a process restart does NOT reset them — a
compromised process cannot delete records and renumber from 1 to evade
gap detection. New `mesh.audit.verify_audit_integrity()` walks the log
and reports `{ok, total, signed, verified, bad_signature, missing_sig,
psk_present, sequence_gaps}`. Bad-signature records do not advance the
per-peer cursor so a tampered seq value cannot mask a real gap on the
next legit record.

### Added: cross-transport deduplication for bridge mode

`mesh.transport.bridge_transport._CommandDeduplicator` collapses the
same command delivered via both Zenoh and AWS IoT MQTT into a single
dispatch. Identity is the envelope nonce when present, otherwise a full
256-bit SHA-256 fingerprint of `(sender_id, turn_id, command)`. TTL via
`STRANDS_MESH_DEDUP_TTL` (default 120 s).

### Added: AWS IoT policy scope tightening

`_ROBOT_POLICY_DOC` and `_OPERATOR_POLICY_DOC` no longer grant
`iot:Receive` on `strands/*`. Each role is restricted to the topics it
actually subscribes to — robots get own `/cmd`, own `/response/*`,
`broadcast`, `safety/estop`, `+/presence`; operators get the monitoring
topics (`+/presence`, `+/state`, `+/health`, `+/safety/event`,
`safety/estop`).

### Added: Amazon Root CA1 SHA-256 pinning + fetch hardening

`_ensure_ca` verifies the pinned SHA-256 fingerprint on every load
(both download and re-use of an existing on-disk copy), caps the
download body at 64 KiB, and times out after 15 s. The public
`verify_ca_pin(path)` helper for ops scripts always does the raw hash
compare and never honours `STRANDS_MESH_DISABLE_CA_PIN` — only the
provisioning-side `_verify_ca_bytes` honours the break-glass.

### Added: `robot_mesh` LLM-tool safety controls

The agent-facing tool now uses `@tool(context=True)` to receive a
Strands SDK `ToolContext`. `emergency_stop` and `broadcast` raise a
proper human-in-the-loop interrupt
(`tool_context.interrupt("robot_mesh-<action>-approval", reason=...)`)
that pauses the agent loop and returns control to the host. Only
canonical affirmatives (`y`/`yes`/`approve`/`approved`,
case-insensitive, whitespace-trimmed) approve. Per-action sliding-
window rate limits (e.g. `emergency_stop` capped at 3/min). Every
safety-significant call is audited via `mesh.audit.log_safety_event`.
Replaces the previous `confirm: bool` parameter.

### Added: 11 new configuration env vars

`STRANDS_MESH_PSK`, `STRANDS_MESH_REQUIRE_AUTH`,
`STRANDS_MESH_REPLAY_WINDOW`, `STRANDS_MESH_POLICY_HOST_ALLOW`,
`STRANDS_MESH_PEER_RATE`, `STRANDS_MESH_AUDIT_PSK`,
`STRANDS_MESH_OVERRIDE_CODE`, `STRANDS_MESH_DEDUP_TTL`,
`STRANDS_MESH_CAMERA_PRESIGN_TTL`, `STRANDS_MESH_CAMERA_DISABLED`,
`STRANDS_MESH_DISABLE_CA_PIN`. All documented in README and in the
`mesh/security.py` module docstring.

### Tests

+143 new tests across:
- `tests/mesh/test_security.py` (58)
- `tests/mesh/test_security_regressions.py` (40)
- `tests/mesh/test_robot_mesh_security.py` (19)
- `tests/mesh/test_audit_integrity.py` (14)
- `tests/mesh/test_bridge_dedup.py` (14)
- `tests/mesh/test_iot_ca_pin.py` (11)
- `tests/mesh/test_iot_policy_scope.py` (7)
- `tests/mesh/test_camera_acl.py` (7)

Total mesh tests: 609 passing (was 440 baseline before #194). No
regressions; permissive-mode back-compat preserves existing zero-config
flows.

### Backwards compatibility

* No PSK configured → permissive mode. Outgoing messages are wrapped
  in an envelope but carry no `sig`; verifiers accept them with a
  one-time WARNING. Bare legacy dicts are still accepted (`v` and
  `payload` keys absent → passthrough).
* Existing API signatures are unchanged except `tools.robot_mesh.robot_mesh`
  lost the `confirm: bool` parameter (replaced by the SDK interrupt).
  Callers that previously passed `confirm=True` now omit it; the
  framework delivers the operator response.
* The fleet-wide e-stop lockout is intentionally soft in permissive
  mode (see README's *Mesh security* section for the rationale).

## Unreleased - #178 (LiberoOffScreenRenderEngine retired)

### Removed: ``LiberoOffScreenRenderEngine`` simulation backend (BREAKING)

After PR #184 made ``MuJoCoSimEngine`` byte-equivalent to upstream LIBERO
(model-level inertias, ``mj_step`` divergence 0 over 200+ substeps, mean
``success_rate=0.92`` vs offscreen ``0.72`` on libero-10/SCENE5),
``LiberoOffScreenRenderEngine`` has no functional reason to exist. It is
deleted entirely.

What is gone:
- **Deleted**: ``strands_robots/simulation/libero_offscreen_render/``
  (entire package, ~700 LoC).
- **Deleted**: ``"libero_offscreen_render"`` registry entry in
  ``strands_robots.simulation.factory`` and its aliases
  ``"libero_offscreen"`` and ``"libero_osr"``.
- **Deleted**: ``LiberoAdapter._on_episode_start_offscreen`` and the
  ``hasattr(sim, "setup_libero_task")`` dispatch branch in
  ``LiberoAdapter.on_episode_start``. The unified ``MuJoCoSimEngine``
  path is the only path now.
- **Deleted**: ``LiberoAdapter.is_success`` no longer delegates to
  ``env.check_success`` on ``OffScreenRenderEnv``-backed engines (no
  such engines exist anymore). It now always evaluates the BDDL
  predicate tree, hardened in #170 / #173 / #175 to match upstream's
  ``check_ontop`` / ``check_contact`` semantics.
- **Deleted**: ``STRANDS_LIBERO_PREDICATE_LOG`` and
  ``STRANDS_LIBERO_PREDICATE_LOG_MAX`` env vars (the BDDL ↔
  ``env.check_success`` disagreement diagnostic; no offscreen env
  to compare against). The ``_walk_predicate_tree`` helper is kept
  for any future BDDL-evaluator debugging.
- **Deleted**: ``tests/simulation/libero_offscreen_render/`` (3 unit
  test files).
- **Rewrote**: ``tests_integ/benchmarks/libero/test_upstream_state_parity.py``'s
  ``test_state_observation_byte_equivalent_at_canonical_init`` to
  compare ``MuJoCoSimEngine`` directly against upstream's raw
  ``OffScreenRenderEnv`` (skipping the intermediate engine wrapper).
  Same coverage, less indirection.

Migration: rename the backend in any ``create_simulation()`` call.

```python
# Before
sim = create_simulation("libero_offscreen_render", ...)
# (also "libero_offscreen", "libero_osr")

# After
sim = create_simulation("mujoco", ...)
```

The ``mujoco`` backend now reaches ``success_rate >= 0.92`` on
libero-10/SCENE5 (vs ``0.72`` for the offscreen engine), so this is
strictly an upgrade for benchmark eval consumers.

Out of scope: ``examples/libero_mujoco.py`` in
``strands-labs/robots-sim`` still has an ``--engine={mujoco,libero_offscreen_render}``
switch. A follow-up issue tracks updating it once this PR lands.

## Unreleased - PR #85 (MuJoCo backend remediation)

### MJCF builder refactor: string-concat -> MjSpec AST (closes #121, #122-#126)

The ``MJCFBuilder`` string-concat path and the ``scene_ops`` XML-round-trip
machinery (~700 lines total) are replaced by direct manipulation of
``mujoco.MjSpec`` - the editable MJCF AST shipped with MuJoCo 3.2+.

What changed under the hood:
- **New module** ``strands_robots/simulation/mujoco/spec_builder.py``. The
  ``SpecBuilder`` class owns scene construction + mutation (``build``,
  ``add_object``, ``remove_body``, ``add_camera``, ``remove_camera``,
  ``attach_robot``, ``from_mjcf_string``, ``from_file``).
- **Deleted**: ``strands_robots/simulation/mujoco/mjcf_builder.py`` (273
  lines of f-string MJCF and the ``_camera_xyaxes_from_target`` helper).
- **Rewrote**: ``strands_robots/simulation/mujoco/scene_ops.py`` from
  ~980 lines of tmpdir + ``mj_saveLastXML`` + ``ElementTree`` round-trips
  down to ~295 lines that go through ``spec.recompile(model, data)``.
- **Bumped**: ``mujoco>=3.0.0`` -> ``>=3.2.0`` in ``pyproject.toml`` so
  ``MjSpec`` is always available. Current hatch env runs 3.8.0.

Agent-visible wins:
- **New action** ``patch_scene_mjcf(ops=[...])`` - apply a list of
  structured ops (add_body, add_geom, add_site, set_body_pos,
  set_body_quat, delete_body) to the live spec atomically. Whole batch
  is rolled back from an XML snapshot if any op fails; one
  ``spec.recompile()`` for the whole batch, so qpos/qvel for unchanged
  joints are preserved. Narrower surface than ``replace_scene_mjcf``
  but much cheaper for surgical edits (no full-scene XML round-trip).
- **New action** ``replace_scene_mjcf(xml=...)`` - atomically replace the
  whole scene with agent-authored MJCF. Validated by actually compiling
  it, so ``<tendon>``, ``<equality>``, ``<pair>``, custom solref/solimp,
  sites, hfield, etc. all work without needing new ``SimObject`` shape
  vocabulary. On malformed XML returns a clean error dict (no process
  abort).
- **``ellipsoid`` shape** now works in ``add_object`` - it's a free
  bonus MuJoCo geom type the string-concat builder rejected.
- **Camera orientation** uses ``quat`` (computed via
  ``mujoco.mju_mat2Quat``) instead of a hand-rolled ``xyaxes`` string.
  Compiled ``cam_mat0`` is numerically identical within ~4e-7.
- **``spec.recompile(model, data)``** preserves existing joint qpos/qvel
  for unchanged joints automatically - no manual "copy state by name"
  loop. Object freejoints added post-compile get initialised to the
  body's ``pos``/``quat``.
- **No more XML injection surface**: names go straight into MjSpec which
  validates them itself, so the old ``_sanitize_name`` regex gate +
  fuzz test are no longer needed.

Downstream API is unchanged: ``add_object``, ``add_robot``, ``remove_object``,
``remove_robot``, ``add_camera``, ``remove_camera``, ``load_scene`` all keep
their tool-action signatures. Tests that asserted on exact XML strings
were rewritten to assert on compiled ``MjModel`` properties (``cam_mat0``,
``mj_name2id``) so they are representation-agnostic.

Known constraint: ``remove_robot`` now rebuilds the scene from scratch
(drops joint qpos state) rather than going through ``spec.delete()`` on
attached bodies. This sidesteps a MuJoCo 3.8 double-free bug where
``spec.delete(attached_body)`` + interpreter shutdown crashes. Trade-off
is documented in ``scene_ops.eject_robot_from_scene``.

### Breaking

These changes tighten the MuJoCo AgentTool contract. Legacy callers that
silently worked by accident will now receive a clear error instead:

- **Router input validation**: The ``_dispatch_action`` router rejects any
  top-level parameter that isn't declared on the target method. Passing
  ``step(num_steps=5)`` (wrong name) or ``set_gravity(device="mps")``
  (stray kwarg) now errors with *"Unknown parameter X for action Y.
  Valid: [...]"* instead of silently dropping the value. Methods whose
  Python signature includes ``**kwargs`` (e.g. ``add_object``) keep their
  pass-through semantics.
- **Missing required args**: produce *"Action X requires parameter Y."*
  instead of a raw Python ``TypeError``.
- **Vector dimension validation**: ``position``, ``target``, ``origin``,
  ``force``, ``torque``, ``gravity``, ``direction``, ``point``, ``orientation``
  (quaternion), and ``color`` (rgba) all validated for length + numeric
  dtype before reaching numpy/MuJoCo.
- **Camera orientation**: ``add_camera(target=[x,y,z])`` is now honoured
  by baking ``xyaxes`` into the MJCF ``<camera>``. Previously the target
  was silently dropped and every custom camera rendered a default view.
  Degenerate case (``target == position``) errors.
- **Render camera validation**: ``render(camera_name="missing")`` errors
  with *"Camera 'missing' not found."* instead of silently falling back
  to the free camera while claiming to render from the named one.
- **Raycast zero-direction guard**: ``raycast(direction=[0,0,0])`` now
  errors with *"direction vector is zero-length"*. Previously MuJoCo's
  C-level ``mj_ray`` would abort the Python process.
- **apply_force requires a non-zero vector**: passing neither ``force``
  nor ``torque`` (or both zero) errors. Previously the call silently
  succeeded with no effect.
- **step(n_steps<0)** rejected (previously it corrupted ``step_count``).
- **Negative mass / timestep / size** rejected per shape; previously
  ``set_body_properties(mass=-1)`` and ``set_timestep(-0.01)`` silently
  succeeded.
- **Plane objects auto-static**: ``add_object(shape="plane")`` now forces
  ``is_static=True`` (planes are infinite in MuJoCo). Explicit
  ``is_static=False`` on a plane is a hard error.
- **Duplicate camera name** rejected. Previously a second ``add_camera``
  with an existing name silently overwrote the registry entry while
  leaving the old camera in the XML - ghost behaviour. Use
  ``remove_camera`` + ``add_camera`` to replace.
- **stop_policy(robot_name='')** errors with *"stop_policy requires
  'robot_name'."* instead of silently matching the first robot.
- **eval_policy** requires an explicit ``robot_name``. Default
  ``n_episodes`` lowered from 10 to 1.
- **register_urdf** validates the path: file must exist, be a file, and
  be readable. Previously bad paths were cached and blew up later.

### Recording backend split

- ``start_recording`` (LeRobotDataset: parquet + per-camera MP4) still
  requires the ``[lerobot]`` extra. Its error message when lerobot is
  missing now points callers at ``start_cameras_recording`` for plain
  MP4 (which runs under ``[sim-mujoco]`` alone via imageio-ffmpeg).
- No API change - the fix is informational.

### Resource hygiene

- ``destroy()`` and ``cleanup()`` now close renderers on the main thread
  and empty the TLS cache. Previously each ``create_world/destroy``
  cycle leaked one ``mujoco.Renderer`` + its GL context (~33 MB per
  cycle measured). Worker-thread renderers still release themselves on
  thread teardown (we avoid cross-thread ``close()`` to prevent
  ``cgl.free()`` SIGSEGVs on macOS).
- ``get_mass_matrix`` and ``get_contacts`` run ``mj_forward`` first so
  values are valid immediately after a ``reset`` or ``add_robot``
  (previously returned stale / uninitialised memory).

### Concurrency guards

Write-mutations are now refused while a policy is running on any robot
in the world. Previously these could race the policy worker thread and
produce undefined behaviour or SIGSEGV:

    reset, set_gravity, set_timestep, set_joint_positions,
    set_joint_velocities, apply_force, set_body_properties,
    set_geom_properties, load_state, randomize, move_object

The error now lists *which* robot(s) are active so the LLM can
``stop_policy`` on each without guessing: *"Cannot 'X' while a policy
is running on 'armA', 'armB'. Stop it first: action='stop_policy'."*

### Concurrent per-robot policies (GH #114)

Multiple ``start_policy`` calls on *different* robots now run
concurrently. MuJoCo physics is still serialized via ``self._lock``
(``mj_step`` and ``ctrl[]`` writes are not thread-safe for concurrent
mutation), but each policy owns a disjoint slice of ``data.ctrl[]`` so
two VLA arms can operate in the same scene without semantic conflict.

- ``start_policy("armA")`` + ``start_policy("armB")`` both succeed.
  Second call no longer hits a global "policy already running" gate.
- ``start_policy`` on the *same* robot while its policy is active
  still errors (unchanged).
- ``remove_robot("X")`` now gracefully stops X's own policy before
  removing, instead of requiring a prior ``stop_policy("X")``. Still
  errors if a *different* robot has an active policy (XML round-trip
  invalidates cached IDs everywhere).
- New action ``list_policies_running`` returns the names of robots
  with live policies. Prunes completed Futures as a side-effect.
- Completed policy Futures are no longer retained forever in
  ``_policy_threads`` (GH #120 companion fix).

### Policy-hook robustness (GH #117)

``PolicyRunner.run`` previously caught *all* ``on_frame`` exceptions at
WARN level and kept iterating. A recording hook with a typo'd observation
key would log 500 lines and produce an empty dataset. Now we count
*consecutive* failures and abort the episode after a threshold (default
5, tunable via new ``max_onframe_failures`` kwarg).

- A single transient failure still logs + continues; counter resets on
  the next successful call.
- ``N`` consecutive failures raise ``RuntimeError`` so ``run()`` returns
  ``status='error'`` with a clear message, preventing silent dataset
  corruption.

### Cleanup graceful shutdown (GH #116)

``Simulation.cleanup()`` no longer races the policy worker. Previously
cleanup set ``self._world = None`` and called ``executor.shutdown(wait=False)``
nearly simultaneously - a policy still inside ``mj_step`` segfaulted on
freed arrays. Now cleanup:

1. Signals every live policy to stop (``policy_running = False``).
2. Awaits each outstanding Future with a bounded timeout (default 5s,
   overridable via new ``cleanup(policy_stop_timeout=...)`` kwarg).
3. Only AFTER workers unwind do we null ``self._world`` and tear down
   renderers / viewer / executor.

Wedged workers that don't stop in time get logged as a warning - cleanup
proceeds rather than hanging the host process on exit.

### Error message consistency

- All "no world" paths return the same string:
  *"No world. Call create_world (or load_scene) first."*
- Unknown-name errors use a uniform ``<Kind> 'X' not found.`` shape
  (Robot / Object / Body / Geom / Joint / Sensor / Camera / Checkpoint).
- ``stop_recording``, ``stop_cameras_recording``, ``stop_policy``,
  ``close_viewer`` are now **idempotent**: calling them when nothing
  is running returns ``status="success"`` with a *"Was not ..."* message
  so callers can invoke them unconditionally.
- ``get_recording_status`` returns success in every lifecycle state
  (no world / not recording / recording).

### Deprecations

- **add_robot name-as-registry fallback**: passing ``name="my_bot"``
  without ``urdf_path`` or ``data_config`` used to resolve ``my_bot`` in
  the model registry. This now fires a ``DeprecationWarning``. Use
  ``add_robot(name="...", data_config="<registry_key>")`` instead. Will
  be removed next major release.

### New / extended actions

- ``forward_kinematics(body_name="X")`` filters to a single body.
- ``get_features(robot_name="X")`` filters to a single robot's joints
  and actuators.
- ``set_geom_properties(geom_name="X")`` accepts the bare object name
  as an alias for the injected ``"{name}_geom"``.
- ``render_all`` flags cameras whose frame has near-zero pixel variance
  (``"⚠️ camera 'X': image appears empty (variance < 1)"``).
- ``render_depth`` surfaces MuJoCo's one-time ``ARB_clip_control``
  warning in the response text on macOS, so the LLM knows when depth
  accuracy is reduced.
- ``render`` / ``render_depth``: width/height validated up front;
  oversized requests get a plain-English message naming the actual
  framebuffer cap (``<global offwidth=...>``) instead of MuJoCo's raw
  error.
- ``run_policy`` / ``start_policy``: accept optional ``n_steps``
  (primary) or legacy ``max_steps`` as an alternative to
  ``duration``+``control_frequency``. ``duration = n_steps /
  control_frequency`` when ``n_steps`` is set.
- **New ``list_policies_running``** action returns the names of robots
  with a live policy - pairs with the new concurrent-policy support
  (see *Concurrent per-robot policies* above).
- ``randomize(randomize_physics=True)`` now reports per-body mass scales
  and per-geom friction scales in the response (not just range
  endpoints).
- ``get_contacts`` resolves unnamed geoms to
  ``"<body_name>/geom_<id>"`` so contact pairs are always human-readable.
- ``get_sensor_data(sensor_name="X")`` on a model with no sensors now
  distinguishes *"Sensor 'X' not found. Model has no sensors."* from
  the generic "no sensors in model" success.

### Tests

- New: ``tests/simulation/mujoco/test_agenttool_contract.py`` - ~50
  tests that lock in router validation, tool_spec ↔ method parity,
  unified error messages, idempotent stop family, ``mj_forward`` before
  reads, render-dim validation, feature filters, camera duplicate
  policy, plane auto-static, policy horizon unification, and more.
- New: ``tests/simulation/mujoco/test_renderer_hygiene.py`` - 4 tests
  asserting TLS cache is emptied on ``destroy``, renderer reuse works
  for identical ``(w,h)``, and ``create_world`` after ``destroy``
  rebuilds cleanly.
- New: ``tests/simulation/mujoco/test_recording_backends.py`` - 2 tests
  (one skipped when ``lerobot`` IS installed) pinning the
  MP4-without-lerobot backend.
- New: ``tests/simulation/mujoco/test_input_validation.py`` - 11 tests
  for step/raycast/apply_force validation.
- New: ``tests_integ/test_resource_hygiene.py`` - 3 integration tests
  (require ``psutil``): 50 create/destroy cycles grow RSS < 50 MB; 500
  renders at fixed dims grow RSS < 100 MB; TLS cache cleared on destroy.

Test count: **256 → 362** (+106 new regression tests), zero
regressions. ``hatch run lint`` (ruff + mypy) clean across 102 source
files.
