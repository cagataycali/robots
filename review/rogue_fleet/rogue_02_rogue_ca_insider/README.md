# `rogue_02_rogue_ca_insider`

**AV-02 — attacker mints a parallel CA chain.**

## Threat narrative

The attacker has gotten further than rogue 01: they understand they
need *some* certificate to handshake against a tls-only listener.
So they stand up their own CA (`openssl req -x509 ...`), mint a leaf
with CN `operator-1` (same convention the real fleet uses), and
attempt to join the bus.

This attack would *succeed* against a deployment that:
* has `enable_mtls = true` but a permissive CA bundle (`/etc/ssl/certs`
  for example), or
* does not enable `verify_name_on_connect`, or
* uses CN-only allowlisting at the application layer rather than
  pinning at the wire.

## What this rogue does

1. Spins up a fresh ephemeral CA in a tempdir (the *attacker's* CA).
2. Mints `CN=operator-1` signed by that rogue CA.
3. Configures Zenoh with `enable_mtls=true`, `verify_name_on_connect=true`,
   and `root_ca_certificate=<rogue CA>`.
4. Connects with `tls/127.0.0.1:<victim-port>`.
5. Tries to publish `{"action": "stop"}` on `strands/victim-r1/cmd`.

The victim's `STRANDS_MESH_TLS_CA` points at *its* CA only; the rogue
leaf does not chain to it. The TLS handshake should fail.

## Defence in scope

* `_zenoh_config.tls_block` sets `enable_mtls: true` and asserts the
  CA file exists & is readable; pins the CA via
  `STRANDS_MESH_TLS_CA`.
* `_zenoh_config.link_protocols_block` denies plain TCP fallback.

## Pass criterion

Either:
* `zenoh.open` raises an `_AlertReceived(UNKNOWN_CA)`-shaped error, or
* the session opens locally but never bridges to the victim and the
  put is dropped at the routing layer.

## How this would fail (regression mode)

* Removing the explicit `root_ca_certificate` field would let Zenoh
  fall back to the system trust store, and many distros ship with
  Let's Encrypt / public roots that are easier for an attacker to
  cross-sign against.
* Disabling `verify_name_on_connect` would accept *any* cert chain
  that validates against *any* trusted root.

## Cross-references

* Unit-style: `pentest_mesh.py::av_02_rogue_ca`.
* Hardening: `_zenoh_config.tls_block`, the `R24-C` patch which
  asserts `STRANDS_MESH_TLS_CA` exists at config-build time.
