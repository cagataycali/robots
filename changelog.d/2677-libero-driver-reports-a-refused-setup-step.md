### Fixed: a refused LIBERO setup step is reported where it refuses

The LIBERO driver's two setup shims read the same `{"status": ...}` envelope
differently: the Isaac shim checked all five of its `Simulation` calls with
inline guards, the MuJoCo shim checked none of its five. A refused
`create_world`, `add_robot` or `load_scene` was therefore discarded, the run
continued on a world that was never set up, and it printed a `success_rate=`
line for a rollout whose setup had failed; a refused
`start_cameras_recording` surfaced later as the zero-frame recording check,
so the recorder never starting read as a camera that produced no frames.
Measured on the real `Simulation`, a missing scene file and a camera the
world does not have produced a byte-identical message, leaving the two
causes indistinguishable.

`_require_ok(envelope, step)` is now the single owner of the rule - it
returns the envelope untouched on success and raises
`RuntimeError("<step> failed: <envelope>")` otherwise - and both shims route
every setup call through it. The Isaac shim's operator-visible wording is
unchanged, and it still chains its `on_frame` handle off the returned
envelope. `sim.destroy()` on the cleanup path and the per-episode
`sim.step()` in the retry loop stay unchecked on purpose, pinned by controls:
a cleanup that refuses must not mask the failure being cleaned up after, and
the step's verdict belongs to the retry loop rather than to setup.
