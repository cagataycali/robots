### Fixed: the test session leaves every `STRANDS_*` environment variable as it found it

`apply_mesh_env` writes `os.environ` directly, so a dashboard test that started a
bridge left `STRANDS_MESH_CAMERA_HZ` set for the rest of its worker and the mesh
roster test found a camera loop it never asked for, once in a few thousand CI
runs. A session fixture now restores the `STRANDS_` prefix after each test, and
the roster test clears the knob itself. Closes #4200.
