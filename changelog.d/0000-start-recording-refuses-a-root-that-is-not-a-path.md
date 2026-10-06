### Fixed: `start_recording` refuses a `root=` that is not a path

`start_recording(root=42)` (or `True`, `b"..."`, a list, a dict, a float)
raised `TypeError` out of the tool instead of returning a
`{"status": "error"}` envelope like `fps=`, `overwrite=` and `repo_id=` do,
and a falsy non-path (`root=0`, `root=False`) was silently ignored and
recorded into the default dataset home. `resolve_dataset_dir` now refuses a
`root` that is neither `None`, a string nor an `os.PathLike`, so every
writer that resolves through it reports the bad value before anything is
armed or overwritten.
