### Tests

- `scripts/check_import_layers.py` grades `strands_robots` against the layered
  import DAG (`core -> registry -> drivers|mesh -> sim|policies -> app -> tools
  -> dashboard`) from source with `ast`: it fails on a cycle in the runtime
  module-scope graph and on an upward import not declared in
  `KNOWN_UPWARD_EDGES`, the shrinking roster of inversions left to fix.
  Typing-only and in-function imports are reported separately, being the
  sanctioned ways to break a cycle. The first inversion it retired:
  `drivers/microduck` derives its 14 locomotion joints from its own 15-joint
  wire map rather than reading them out of `policies/microduck`.
