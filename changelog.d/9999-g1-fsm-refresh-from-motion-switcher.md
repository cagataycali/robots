### Added: `G1Driver` reads its FSM id from the motion-switcher API

Previously `G1Driver._fsm_id` had exactly one writer — the `None` initialiser
in `__init__` — so `_check_motion_gates` refused every `send_action`,
`run_policy` and `start_task` with `FSM id unknown - motion-switcher source
has not been wired`. The wire path the SDK exposes for the FSM state is
`MotionSwitcherClient.CheckMode()` (`LowState_` has no `fsm_id` field; every
SDK example uses the motion switcher's `CheckMode` return under `form`);
issue #2765 collected the wire-format decisions that read shape depends on,
and `strands_robots/tools/g1/_motion_switcher.py` landed the decoder half
in that thread.

The driver-side wiring lands here as the producer. `G1Driver` takes a new
keyword-only `motion_switcher_client_factory` constructor argument — a
callable that returns an open `MotionSwitcherClient`, defaulting to a lazy
loader that imports the SDK on first call — and a private `_refresh_fsm_id`
method that reads through `read_fsm_id` at the top of every
`_check_motion_gates` call. Three read-side branches are handled
explicitly: an OK reading writes `_fsm_id`, a `name == ""` reading (the
SDK's "no motion mode selected" state) clears the cache so a stale value
from before `ReleaseMode()` does not silently open the gate, and a refused
reading leaves `_fsm_id` at its previous value so a transient
`CheckMode()` failure on the tenth step of a rollout does not clobber the
id the previous nine wrote.

`get_status` now surfaces the mode label, the last decoder refusal, and
the factory open error alongside the integer id, so a caller inspecting
the mesh peer sees the same information the gate reads. The default lazy
loader preserves the module-load hygiene invariant every G1 module
already carries: `unitree_sdk2py` is imported inside function bodies,
never at module load, so the driver still imports on Thor, on CI, and in
every unit test with a mocked bus.

The predecessor un-reachability test file (`test_g1_battery_floor_is_gated_behind_the_unwired_fsm.py`)
is replaced by `test_g1_battery_floor_reaches_with_wired_fsm.py`, which
grades the flipped reachability directly: a driver with a wired FSM and a
critical pack refuses for the *battery*, not the FSM. The acceptance test
`test_send_action_returns_success_on_a_healthy_driver_that_has_a_decoded_lowstate`
turns over from `strict=True` XFAIL into a passing cell — the same
mechanical checkpoint the predecessor's docstring promised the wiring
commit would fire.

Closes the driver-side half of harness#361 and issue #2765's "decidable
now" list. What still needs a G1 in the room: measuring which `form`
integer the firmware reports for `HANDSHAKE_FSMS` under `CheckMode`, and
whether `ReleaseMode()` is required before an `rt/lowcmd` write (the SDK's
own G1 low-level example calls it, and this PR does not).
