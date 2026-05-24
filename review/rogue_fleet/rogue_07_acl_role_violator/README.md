# `rogue_07_acl_role_violator`

**AV-06 / AV-07 / AV-08 — ACL shape gate.**

## Threat narrative

The ACL file is the **only** wire-layer authorisation mechanism in
the defended posture. A subtle shape mistake in operator config
silently degrades it from "role separation" to "any CA-signed peer
may publish anywhere". Two such mistakes are common in the wild:

1. `enabled: false` — a debug toggle left from local testing.
   Zenoh treats the entire ACL block as a no-op.
2. `interfaces: []` — an empty list under each subject. Zenoh's
   subject-matching code treats this as "matches no peer", so every
   rule fires for nobody.

The **F2 fix** moves these from "silent degradation" to "loud
refusal at config-build time".

The F2 happy path also accepts the cleanest ACL pattern: CN-only
(no `interfaces` field, which Zenoh treats as wildcard — the right
default for CN-gated role separation).

## What this rogue does

Directly invokes `_acl_config._validate_acl_shape` against three
hand-built ACL dicts:

* `bad1`: `enabled: false` → must raise
* `bad2`: `interfaces: []` → must raise
* `good`: no `interfaces` field, just `cert_common_names` → must
  NOT raise

## Defences in scope

* `_acl_config._validate_acl_shape` (the F2 patch) — fail-closed
  on the two known silent-degrade shapes.
* Doc claim relaxation: PR #195 audit `§2.2` flagged that requiring
  `interfaces` is wrong; F2 honors that and lets the field be
  omitted entirely.

## Pass criterion

All three sub-tests have the expected outcome.

## Failure modes

* If the validator regressed to accepting `enabled: false`,
  operators silently ship insecure ACLs.
* If the validator regressed to *requiring* `interfaces`, the
  cleanest CN-only deployment posture is rejected and operators
  work around with placeholder values that may not match reality.

## Cross-references

* Unit-style: `pentest_mesh.py::av_06_acl_enabled_false`,
  `av_07_acl_empty_interfaces`, `av_08_acl_cn_only_accepted`,
  `av_34_shipped_acl_example_loads`.
* Source: `_acl_config._validate_acl_shape`.
* Audit: `PR195_zenoh_security_audit.md §2.2`.
