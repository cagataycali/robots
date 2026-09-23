### Fixed: the dashboard settings test store leaves the environment as it found it

`settings.apply_mesh_env()` writes `os.environ` by design, and the scratch-store
fixture in `tests/test_dashboard_settings_value_domain.py` dropped each declared
variable with `monkeypatch.delenv(name, raising=False)` -- which records nothing
to undo when the variable is absent. A published `ZENOH_CONNECT` therefore
outlived its test and reached an unrelated mesh session as that key's default,
which refused to start over a `tcp` scheme under `STRANDS_MESH_AUTH_MODE=mtls`.
The fixture now records the pre-state before dropping, so teardown removes what
was published and restores what the caller's environment carried.
