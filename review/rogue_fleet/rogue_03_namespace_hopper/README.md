# `rogue_03_namespace_hopper`

**AV-03 — cross-fleet routing isolation.**

## Threat narrative

Two strands fleets coexist on the same LAN: `service-robots` (fleet A,
namespace `strands-svc`) and `warehouse-agvs` (fleet B, namespace
`strands-agv`). An attacker who has compromised one peer in fleet A
should not, by virtue of bus reachability, be able to drive any robot
in fleet B.

This matters for blast-radius containment after a single peer is
compromised. Without namespace isolation, mTLS alone would let any
CA-trusted peer publish anywhere on the bus.

## What this rogue does

1. Opens a peer with **valid mTLS material** (signed by the same CA
   the victim trusts).
2. Uses `namespace = "other"` (the victim is `"strands"`).
3. Subscribes to `strands/**` to verify it cannot see the victim's
   presence heartbeats.
4. Publishes a stop command at `victim-r1/cmd` (the wire key becomes
   `other/victim-r1/cmd` after Zenoh's egress namespace prefix).

## Defence in scope

* `_zenoh_config.namespace_block` -- sets `namespace` field; Zenoh
  prepends it to every put on egress and strips it on ingress.
* `_acl_config.acl_block` (when an ACL file is loaded) -- the ACL
  rules `key_exprs` operate on the post-strip key, so even if the
  namespace check failed, role separation still applies.

See:
* `eclipse-zenoh/zenoh/src/net/routing/namespace.rs:155-189` -- ingress
  strip, plus 45-55 -- egress prepend.
* CHANGELOG section 8: "Namespace isolation: a strict, transport-level
  fleet boundary that operates *before* ACL."

## Pass criterion

The rogue saw **zero** samples on its `strands/**` subscriber. That
means the namespace boundary is symmetric: we cannot read into the
victim's namespace, and our cmd put cannot bridge into it either.

## Failure modes

* If `namespace_block` did not insert the namespace field, both
  fleets would publish into the same key space and the rogue's
  `victim-r1/cmd` would land at the victim.
* If the victim's subscriber were declared at `**/cmd` (no
  namespace prefix), even with namespace insertion the victim
  would receive cross-fleet messages.

## Cross-references

* Unit-style: `pentest_mesh.py::av_03_namespace_isolation`.
* Source: `strands_robots/mesh/_zenoh_config.py::namespace_block`.
