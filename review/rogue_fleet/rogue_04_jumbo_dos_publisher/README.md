# `rogue_04_jumbo_dos_publisher`

**AV-04 — jumbo-frame DoS at the receiver deserialiser.**

## Threat narrative

A compromised peer (or a misbehaving robot with a stuck JSON
serialiser) starts shoving multi-megabyte payloads onto the cmd
topic. Without a transport-level size cap, every receiver in the
fleet pulls the bytes off the wire, allocates a JSON parser, and
blocks while it deserialises. A handful of these in-flight is enough
to OOM a Raspberry-Pi-class robot.

This is precisely the threat the **F1 fix** addressed: the
`low_pass_filter` block previously enumerated NIC names with a hard-
coded fallback list (`["lo", "lo0", "eth0", "en0", "en1", "wlan0"]`).
On hosts whose NIC was named anything else (`enp0s3`, `wlP1p1s0`,
`cni0`, `wg0`), the cap silently bound to nothing, and jumbo cmds
bypassed the filter. Post-F1, the `interfaces` field is omitted
(Zenoh treats `None` as wildcard, applying the cap to every link).

## What this rogue does

1. Opens a peer with valid mTLS material (operator cert).
2. Configures **no** `low_pass_filter` of its own — the publisher
   side has no transmit cap, only the receiver does.
3. Publishes a baseline 50-byte `status` cmd.
4. Publishes a 32 KiB jumbo cmd (`STRANDS_MESH_MAX_CMD_BYTES=512`
   on the victim, so 64x the cap).
5. Sends a status probe and listens for a reply to verify the
   victim is alive.

The receiver-side `low_pass_filter` drops jumbo payloads silently
before they reach the deserialiser. From a publisher's perspective
this is invisible — there is no NACK — so the test asserts
**liveness** (the victim survives) rather than direct rejection.

## Defence in scope

* `_zenoh_config.low_pass_filter_block` (post-F1):
  ```python
  [{
    "messages": ["put"],
    "flows": ["ingress", "egress"],
    "key_exprs": ["**/cmd", "**/broadcast"],
    "size_limit": <bytes>,
  }]  # NO `interfaces` field -> wildcard binding
  ```
* `STRANDS_MESH_MAX_CMD_BYTES` env var sets the cap.

See F1 commit (PR #195) and `pentest_mesh.py::av_05_jumbo_cmd_unusual_iface`.

## Pass criterion

The victim is **still responsive** to a follow-up status probe after
the jumbo. We cannot directly observe the drop from the publisher
side; the indirect liveness test is the strongest signal a process-
isolated attacker has.

For a stronger assertion, the orchestrator could parse the victim's
audit log post-run; future enhancement (`reach: --inspect-audit`).

## Failure modes

* If the victim's `low_pass_filter_block` regressed to enumerating
  NICs, and the test host has an unusual NIC name, the cap binds
  to nothing and the jumbo is processed: the victim might OOM
  silently and miss the status probe.

## Cross-references

* Unit-style: `pentest_mesh.py::av_04_jumbo_cmd`,
  `pentest_mesh.py::av_05_jumbo_cmd_unusual_iface`.
* F1 audit: `PR195_zenoh_security_audit.md §2.3`.
