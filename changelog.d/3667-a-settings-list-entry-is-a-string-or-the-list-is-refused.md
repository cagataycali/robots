### Fixed: a dashboard settings list entry is a string or the list is refused

`settings._coerce_strict` graded a list key's container shape ("expected a
list or comma-separated string") and not its entries, and `_as_list` spelled
each entry with `str()` on its way to the store. A `null` inside an otherwise
well-formed `mesh.connect` list on the strict path (UI / API,
`update_strict`) was therefore accepted, stored as the endpoint `"None"`, and
published by `apply_mesh_env()` as `ZENOH_CONNECT=None`; `[1, 2]` reached the
wire as `1,2`, a nested list as its `repr`, and `security.cors_origins` took an
origin `"None"` by the same route. An entry is now a string or the list is
refused, naming the entry index and the type received; the lenient path
(file / environment / CLI) degrades to `[]`, the key's own shape, as it does
for every other unusable value. Pinned by
`tests/test_dashboard_settings_value_domain.py`, which also holds that the
environment after a refused write carries no endpoint and that a list of
strings is still held whole.
