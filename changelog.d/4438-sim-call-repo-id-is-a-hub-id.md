### Fixed: a `sim_call` `repo_id` is a Hub id on the wire

A mesh peer could name a directory on the recording host as `repo_id` (a `/` or `./` prefix reads as a verbatim directory) and, with `overwrite`, have `start_recording` remove it. The wire now admits the `owner/name` shape only and refuses anything else with the rule in the sentence.
