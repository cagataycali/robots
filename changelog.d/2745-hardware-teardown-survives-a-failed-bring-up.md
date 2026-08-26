### Fixed: teardown after a device bring-up that failed reports nothing, instead of naming an internal attribute

`Robot(..., mode="real")` builds its lerobot device in one statement,
`self.robot = self._initialize_robot(...)`, so when that statement raises the attribute it assigns
never comes into existence. An absent motor-SDK extra is the ordinary way to get there, and the
message naming the extra is the whole value of that path. `cleanup()` then runs from `__del__`,
reaches `_disconnect_devices` -> `_close_open_devices`, and both read the device handle as
`getattr(self.robot, "bus", None)` - which guards the *inner* attribute while dereferencing
`self.robot` itself. Teardown raised `AttributeError: 'Robot' object has no attribute 'robot'`,
`cleanup()` caught it, and the operator was shown a library-internal attribute name beside the
install hint they actually needed. Measured on `Robot("koch", mode="real")` with the `dynamixel`
extra absent: one `ERROR Cleanup error for koch: 'Robot' object has no attribute 'robot'` ahead of
the actionable `'dynamixel-sdk' is required but not installed`, and none after.

`_close_open_devices` is the method this is most surprising in, because its own docstring names "a
failed connect" as one of its two callers - it exists for a device set that is only partly open, and
could not survive the most partial failure there is. Both readers now bind the handle with
`getattr(self, "robot", None)`, which is how the sibling teardown reader `_shutdown_ros_bridge`
already reads its own handle, so this is that spelling applied consistently rather than a new
convention. Reading it tolerantly rather than skipping teardown when it is absent matters because
`_close_open_devices` has three callers besides `cleanup()`, on the failed-connect rollback path;
gating only the `cleanup()` call site leaves those three unprotected.

The invariant `__init__` already claims - its mesh attributes are set before the bring-up "so
`cleanup()`/`__del__` never see an `AttributeError` if construction fails partway through" - is now
graded against the attributes teardown actually reads, derived from the call closure rather than
from a list, so an attribute assigned after the bring-up and read during teardown fails whether or
not anyone remembers to add it. Assignments are credited across the closure, because
`self.<attr> = ...` creates state that outlives the method doing it and is how
`TeleopMixin._ensure_teleop_state` protects the siblings its caller invokes afterwards; tolerant
probes are credited only to the method that makes them, because `getattr(self, name, default)` binds
a local, and crediting it closure-wide would let one reader's guard excuse a sibling's bare
dereference - which is this defect exactly.

The existing guard for this failure,
`test_mesh_attrs_set_before_initialize_robot_no_attribute_error_in_cleanup`, could not fail. It
selected records with `"AttributeError" in r.message`, and `cleanup()` logs
`f"Cleanup error for {name}: {e}"`, which interpolates `str(e)` - the message alone, never the type
name - so its `offenders` list was empty whatever the code did. It also asserted before the
finalizer had run: the exception's traceback keeps the half-built instance alive, so `__del__` had
not yet been reached when the `pytest.raises` block exited. It passed while printing this defect on
its own output line. Both are fixed in place, and the pair is graded: reverting the mesh hoisting it
guards now fails it, and reverting that hoisting with the forced collection removed does not.
