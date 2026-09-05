### Fixed

- **dashboard/auth**: the two caps on the pending-WebAuthn-challenge table,
  `STRANDS_DASH_AUTH_CHAL_MAX` and `STRANDS_DASH_AUTH_CHAL_MAX_PER_IP`, are now read
  through one domain that refuses a pair which cannot deliver the per-client fairness
  the per-ip cap exists for. That fairness is a relation between the two caps rather
  than a range on either alone: the per-ip cap only binds while it is the smaller of
  the two, and the global cap's eviction is ip-blind. Measured over a 10x10 grid of
  pairs, 55 of 100 previously accepted pairs let one client evict the operator's
  pending login - including a narrowed `CHAL_MAX=8` left beside the default per-ip 16.
  A non-integer setting now names the variable instead of raising a bare `int()`
  `ValueError`, and an empty setting means unset, as it already did for the TTL
  readers next door. Regression tests pin the refused pairs, the narrowest usable pair
  (`CHAL_MAX=2`, `CHAL_MAX_PER_IP=1`), and the flood outcome that makes the relation
  load-bearing.
