### Tests: the prune's stand-in has to sit where the verdict is answered

`SessionManager._load_sessions` decides whether a teleop session's process is
gone through `_process_stop.session_is_running`, which resolves `psutil` from
its own module's globals. One case controlled the prune by rebinding
`lerobot_teleoperate.psutil`, which that path does not read: the stand-in was
consulted zero times and the verdict fell through to the real machine. With the
record naming a hard-coded pid 4242 the case passed on a host where that pid
was free and failed on one where it was taken, having graded neither -- and
removing the `except psutil.NoSuchProcess: return False` handler its docstring
named leaves it green, so it pinned nothing.

The race it claimed is already pinned by
`test_a_process_reaped_mid_probe_is_still_pruned`, which names a live pid,
carries the recorded identity that makes the second probe reachable at all, and
doubles both seams the verdict reads (the shared `psutil` object and the procfs
identity read). The superseded case and its stand-in are removed, and the seam
is pinned in the module that owns it: a census that no test controls a process
probe by rebinding the name `psutil`, the measurement that such a rebinding is
not consulted by the prune, and the control that setting the attribute on the
module object is. The rule is scoped to that one name because rebinding a whole
module in another module's globals is the right tool where that module is the
reader, and usually is: 20 of the 21 such rebindings in the tree are sound, 15 of
them installing a fake clock.
