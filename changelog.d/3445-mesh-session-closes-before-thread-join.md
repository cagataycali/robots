### Fixed: a process that joined the mesh exits on its own

`atexit` hooks run after `threading._shutdown()` has joined every non-daemon
thread, and zenoh serves each subscriber callback from a non-daemon thread that
ends only when its session closes. The mesh session singleton's teardown was
registered with `atexit`, so it could never get its turn while any peer still
held the session: the interpreter waited on the callback threads, which waited
on the close. The first `docs/mesh.md` example - a `Robot(..., mesh=True)`,
whose child SimRobot peer holds the session even after `mesh.stop()` closes the
root peer - therefore never exited, and neither did a robot-less process that
reached the fleet through the `robot_mesh` tool's gateway. Only an explicit
`cleanup()` got out.

The teardown now registers with `threading._register_atexit`, the pre-join hook
`concurrent.futures` uses for the same reason, falling back to
`atexit.register` where that CPython-private name is absent.
`_stop_gateway_mesh` stays on `atexit` and is correct there: it does not own
the session, and every thread its own `Mesh.stop` ends is a daemon.
