### Fixed: a registry-stored asset path is contained wherever it is joined

`model_xml` and `scene_xml` are stored relative to a robot's asset directory and
every reader joins them back through `safe_join`, but three writers joined them
raw and so answered questions about files the readers never open:

- `register_robot` validated existence with `resolved_dir / model_xml`, and an
  absolute value discards `resolved_dir` entirely - any existing host file
  satisfied the check, and the deferred `add_robot()` failure that check exists
  to prevent came back with the entry already persisted. Such a value is now
  refused at registration, naming which of the two paths to correct.
- `_needs_download` read `search_dir / asset_dir / xml_file`, so a readable
  model outside every search path made a robot report as present and needing no
  fetch while the resolver found nothing.
- `_download_via_robot_descriptions` validated a freshly linked directory with
  `dst / model_xml`, so an absolute value reported `downloaded` for a link that
  holds no model at all.
