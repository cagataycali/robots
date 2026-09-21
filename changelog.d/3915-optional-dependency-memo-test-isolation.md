### Fixed: a stand-in for an absent optional dependency no longer outlives its test

`require_optional` memoises every optional dependency it resolves in a
process-global dict. A test stands in for a module nothing installs by
rebinding `sys.modules`, and `monkeypatch` restores that binding - but not the
copy the package took, so the stand-in was served to every later caller in the
process. With `tests/policies/moveit2/test_zmq_sidecar.py` ahead of the
GR00T/MoveIt2 client files (the ordering `--dist loadfile` produces, not a
serial run), 41 cells died on `'types.SimpleNamespace' object is not callable`.
The session restores the memo now, beside the sibling teardowns for the Device
Connect module swap and the mesh rate-limit window; entries a test installs into
the memo itself stay monkeypatch's to undo.
