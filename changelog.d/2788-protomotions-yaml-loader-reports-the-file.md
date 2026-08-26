### Fixed

- **policies/protomotions**: `load_config_from_yaml` now reports a yaml sidecar
  it cannot use the way its three sibling policy-config file loaders already do.
  It is the fourth such loader and carried none of the four guards the other
  three share, so the failure a caller saw named a method of the parsed value
  rather than the file. The case that decides it is an **empty** sidecar - a
  truncated download, a `touch`ed placeholder, a file holding only comments:
  `yaml.safe_load("")` is `None`, so the first field lookup raised
  `AttributeError: 'NoneType' object has no attribute 'get'`, while an empty
  mapping `{}` - the same information, every field absent - already returned the
  all-defaults config this function documents absent fields to mean ("a missing
  block is not an error"). Two spellings of one input, and one of them
  dead-ended; both now load the defaults. Alongside that, `~` is expanded (a
  sidecar at `~/unified_pipeline.yaml` was reported *missing* while it existed,
  quoting the literal `~`), the path must name a file rather than merely exist
  (a directory reached the read and surfaced as `IsADirectoryError`), malformed
  yaml is wrapped into the documented `ValueError` naming the path instead of
  escaping as a bare `yaml.YAMLError`, and a document holding a list or a scalar
  is refused by type (`ProtoMotions yaml /tmp/s.yaml must contain a mapping, got
  list`) instead of reaching `data.get(...)`. `KimodoConfig.from_json`,
  `MotionBricksConfig.from_file` and `WBCConfig.from_file` shipped all four
  already, so this is three of four loaders' behaviour applied to the fourth.
  Reachable through the public `ProtoMotionsPolicy(yaml_path=...)` constructor,
  which forwards a caller's path straight in. The extension stays unchecked, for
  the reason `from_json` documents for its own: a yaml document stored under any
  name loads today, and refusing it would stop a payload that currently works.
  The new tests derive the rule over *reading a path from disk* rather than over
  the `from_*` classmethod shape the existing guard keys on - which is why a
  module-level loader could land unheld - so all four are graded however they
  are spelled.
