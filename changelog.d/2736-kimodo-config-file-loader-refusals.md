### Fixed

- **policies/kimodo**: `KimodoConfig.from_json` now reports a config file it
  cannot read the way its two sibling policy-config loaders already do. It
  parsed the file and handed the result straight to `from_dict`, whose first act
  is `data.items()`, so a file holding any JSON value other than an object
  surfaced as `AttributeError: 'list' object has no attribute 'items'` - a
  message naming a method of the parsed value rather than the file that could
  not supply fields. Measured across the five non-object JSON values, all five
  failed that way; all five are now `ValueError` naming the class, the resolved
  path and the type found (`KimodoConfig file /tmp/cfg.json must contain a
  mapping, got list`). A file that was not JSON escaped as a bare
  `json.JSONDecodeError` and is now wrapped with the file named, a path that is
  not a file raises `FileNotFoundError` naming the class instead of an errno
  alone, and `~` is expanded - a config at `~/kimodo.json` was previously
  reported missing while it existed, with the literal `~` in the message.
  `MotionBricksConfig.from_file` and `WBCConfig.from_file` shipped all four
  already, so this is two of three loaders' behaviour applied to the third, and
  the new tests grade the rule over all three by discovering them rather than
  listing them. The extension is deliberately still unchecked, unlike those two:
  a JSON object stored under another name loads today, and refusing one would
  stop a payload that currently works, so no input that loads today stops
  loading.

- **policies/kimodo**: the unrecognised-key policy is now documented as the one
  the code implements. `from_dict` described the drop as happening "with a
  warning"; measured, none of the three policy configs warns, and the other two
  document the drop as silent forward compatibility, so the docstring was the
  outlier. `docs/policies/kimodo.md` presents three interchangeable ways to set
  a field and claimed a misspelled knob "raises `TypeError` at construction
  instead of being silently ignored" - true for the two keyword forms, and not
  for the `config` dict, which is read by `from_dict`:
  `KimodoPolicy(config={"diffusion_stpes": 25})` builds with the default 100 and
  emits nothing. The page now says which form refuses and which drops, and how
  to have a typo refused.
