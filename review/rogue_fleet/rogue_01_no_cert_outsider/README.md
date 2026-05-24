# `rogue_01_no_cert_outsider`

**AV-01 — LAN attacker without PKI material.**

## Threat narrative

An attacker has L2/L3 reach to the same network the victim robot is on —
a rogue laptop on the warehouse Wi-Fi, a compromised camera on the
robot VLAN, a malicious container co-tenanted with the victim. They
have **no certificate the victim's CA trusts**.

Goal: get a `cmd` onto `strands/<victim>/cmd` (the dispatch topic) so
the victim's `Mesh._on_cmd` runs an attacker-chosen action.

## What this rogue does

It opens a `zenoh.Session` with:

* `mode = "peer"`
* `connect/endpoints = ["tcp/127.0.0.1:<victim-port>"]` (plain TCP, no TLS)
* No CA, no client cert, no key.

Then it calls `session.put("strands/victim-r1/cmd", "<malicious payload>")`
and waits for the routing graph to converge.

## Defence in scope

Two Zenoh-native defences gate this:

1. **`transport/link/protocols = ["tls"]`** —
   Source: `strands_robots/mesh/_zenoh_config.py::link_protocols_block`.
   The victim only advertises `tls/...` listen endpoints. A plain-TCP
   connect to the same port sees an immediate handshake failure on the
   first byte (Zenoh's TLS terminator does not fall through to plain
   TCP).

2. **`transport/link/tls.enable_mtls = true` + `verify_name_on_connect = true`** —
   Source: `strands_robots/mesh/_zenoh_config.py::tls_block`.
   Even if the attacker switched to `tls/...`, they have no client cert
   so the TLS handshake fails before any application byte flows.

## Pass criterion

The attack is *held* when, after the rogue completes its publish, the
victim's audit log contains **no** `command_received` event with the
rogue's payload. Cross-process inspection of the audit log is brittle
from a separate pid (PSK / file mode), so we use the link-layer
proxy: a `tls`-only listener does not accept plain-TCP peers; the
routing graph never bridges them. If `zenoh.open` raises, that's the
strongest signal; if it succeeds but no link establishes, the put is a
no-op.

## How this would fail (if the PR regressed)

* If `link_protocols_block` ever regressed to allowing both `tcp` and
  `tls`, the plain-TCP rogue's session would join the graph and the
  put would arrive at the victim, where `_on_cmd` would dispatch.
* If `enable_mtls` was off, a self-signed client cert would suffice.
* If `verify_name_on_connect` was off, a self-issued cert with a
  mismatched CN would be accepted.

## Cross-references

* In-process unit-style version: `pentest_mesh.py::av_01_no_cert_outsider`.
* Source: `strands_robots/mesh/_zenoh_config.py` `tls_block` and
  `link_protocols_block`.
