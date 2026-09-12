### Fixed: a Device Connect bring-up that returns no runtime is reported, not handed back

`init_device_connect_sync` promises a `DeviceRuntime`, and released the loop it
created when the bring-up *raised* (#3530). A bring-up that finished without
returning a runtime left the same empty holder, but took neither path: the
wrapper returned `None` against its own annotation, and the bring-up thread
parked in `run_forever` on a loop nothing would ever serve on and nothing could
reach, since the loop and thread are handed to the caller only by being adopted
onto the returned runtime. Measured, six such attempts left six parked threads,
six open loops and eighteen descriptors, exactly the leak the raising route no
longer has.

An empty holder is now a failed bring-up however it came to be empty: the
release gate on the bring-up thread asks whether a runtime came up rather than
whether an exception was recorded, and the wrapper raises `RuntimeError` naming
the missing runtime. That also restores the operator's status line - the
foreground runner reports a missing runtime as "see the warning above", and on
this route nothing had logged one.
