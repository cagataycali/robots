# `rogue_14_iot_bootstrap_mitm`

**AV-IOT-CA-PIN — rogue CA at IoT bootstrap.**

## Threat narrative

Fleets that bootstrap through AWS IoT pull `AmazonRootCA1.pem` over
HTTPS during first-run. A network-level adversary at any point on
that path (DNS hijack, captive portal, BGP route attack, malicious
corporate proxy that re-signs everything) can return a rogue CA
bundle. Without a pinned hash check, the bootstrap silently
trusts the rogue chain — and every subsequent IoT call rides on
an attacker-issued cert.

Variants the attacker would try:

1. **Plant a rogue CA on disk** before bootstrap runs and rely on
   the fast-path code re-using the file without re-checking.
2. Set `STRANDS_MESH_DISABLE_CA_PIN=true` (the documented break-
   glass for legitimate re-signing proxies) and hope it's broad
   enough to also disable the existing-file pin check.
3. Inject malformed pins via `STRANDS_MESH_CA_PINS` to confuse the
   parser and silently fail-open.

## Defences in scope

* `provision._AMAZON_ROOT_CA1_PINS` — built-in canonical SHA-256
  pin set (64-char lowercase hex).
* `provision._verify_ca_bytes` / `_hash_matches_pin` — the actual
  pin check.
* `provision._download_or_verify_amazon_root_ca` — the existing-
  file branch ALWAYS pins, regardless of `STRANDS_MESH_DISABLE_CA_PIN`.
  The break-glass is download-only.
* `provision._resolve_ca_pins` — charset-validates each operator
  pin; malformed entries dropped with WARNING (fail-loud, not
  fail-open).

## What this rogue does

6 sub-checks:

1. Built-in pin set non-empty.
2. All built-in pins are 64-char lowercase hex.
3. Rogue bytes fail `_verify_ca_bytes`.
4. Rogue bytes fail `_hash_matches_pin`.
5. Existing-file branch rejects rogue CA even with
   `STRANDS_MESH_DISABLE_CA_PIN=true`.
6. Operator extension `STRANDS_MESH_CA_PINS` accepts well-formed
   hex hashes and drops malformed entries.

## Pass criterion

All 6 hold.

## Cross-references

* Source: `strands_robots/mesh/iot/provision.py` — `_resolve_ca_pins`,
  `_verify_ca_bytes`, `_download_or_verify_amazon_root_ca`.
* Review history: F3 (charset gate), F-series IoT hardening.
