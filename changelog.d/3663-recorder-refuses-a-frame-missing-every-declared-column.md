### Fixed: `DatasetRecorder.add_frame` refuses a frame missing every declared column, not only one

The direct-API default (`required_action_keys=None`) documents that a declared
state column absent from the observation, or a declared action column absent
from the action, raises `ValueError`. Both guards were reached only through a
non-empty dict - the state door sat inside `if state_keys:`, the action door
widened its scope only `if action:` - so a frame that omitted one column was
refused while a frame that omitted all of them (an empty action, or a
camera-only observation against a schema that declares joints) walked past and
reached the dataset with the column missing. What the caller then saw was the
write's own refusal - a `RecordingFrameError` naming the dataset column, or
with `strict=False` a frame dropped and counted under a warning - instead of
the documented `ValueError` naming the columns and the remedy.

The declared state columns are now resolved from the schema before the
`if state_keys:` guard and graded whenever the unscoped default applies, and
the action default requires every declared column unconditionally. The
`sorted()` fallbacks still run only for a non-empty dict, so neither door can
cache an empty column list for the episode. The backends' recording hooks,
which always pass an explicit scope, are unchanged.
