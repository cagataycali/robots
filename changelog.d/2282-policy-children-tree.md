### Added

- `Policy.children` declares the policies a wrapper delegates to, and
  `iter_policy_tree` walks that tree, so a runtime capability probe can reach the
  concrete policy inside a wrapper instead of type-testing the wrapper.
