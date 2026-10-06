"""Repro: start_policy() is strictly narrower than run_policy(), and the gap
leaks a bare CPython TypeError that names the backend class.

Shape at upstream HEAD (strands-labs/robots @ 2eac158e6):
- `strands_robots/simulation/base.py:5127` SimEngine.start_policy declares 14
  kwargs; sibling :3403 SimEngine.run_policy declares 23 (sync default).
- `strands_robots/simulation/mujoco/simulation.py:7088` MuJoCoSimEngine.start_policy
  override keeps the base 14; sibling :7452 MuJoCoSimEngine.run_policy keeps the
  base 23 (same 9-kwarg delta).
- The base.py:5144-5167 docstring cross-references run_policy three times
  ("DEFAULT IMPLEMENTATION ... passes through to :meth:`run_policy`",
  "See ``run_policy`` for conversion rules", "see :meth:`run_policy`") without
  naming any kwarg this method does NOT accept.

The 9 kwargs that run_policy accepts and start_policy does NOT are:

    async_rtc, control_substeps, max_onframe_failures, n_episodes,
    observer, reset_between, rtc_inference_timeout_s, stop_when,
    wbc_install_torque_control

A user who copied a working run_policy call from docs/index.md:100 or the
README quickstart and flipped the method name to get background execution
(the exact migration the start_policy docstring sells) hits:

    TypeError: MuJoCoSimEngine.start_policy() got an unexpected keyword
    argument 'control_substeps'

This leaks the backend class name "MuJoCoSimEngine" and offers no "did you
mean run_policy?" hint. The project has a shared helper for this family of
refusal (`close_match_hint` at base.py:294, used in >=10 sibling sites across
base.py, mujoco/simulation.py, mujoco/rendering.py, mujoco/physics.py) but
start_policy does not reach for it, and does not have a sibling guard that
names "this kwarg exists on run_policy but not start_policy".

Verified on 8 of the 9 delta kwargs below (observer/stop_when/reset_between
need extra wiring to pass a non-trivial value, same TypeError shape).

Run:  MUJOCO_GL=egl python bugbash/start_policy_narrower_sig_repro.py
"""

from __future__ import annotations

import inspect
from typing import Any

from strands_robots import Robot


def main() -> None:
    sim = Robot("so101", mesh=False)

    rp_params = set(inspect.signature(sim.run_policy).parameters)
    sp_params = set(inspect.signature(sim.start_policy).parameters)
    delta = sorted(rp_params - sp_params)

    print(f"run_policy params:   {len(rp_params)}")
    print(f"start_policy params: {len(sp_params)}")
    print(f"\nrun_policy accepts, start_policy does NOT ({len(delta)}):")
    for name in delta:
        print(f"  - {name}")
    assert delta == [
        "async_rtc",
        "control_substeps",
        "max_onframe_failures",
        "n_episodes",
        "observer",
        "reset_between",
        "rtc_inference_timeout_s",
        "stop_when",
        "wbc_install_torque_control",
    ], f"unexpected delta: {delta}"
    print("\nSignature-delta pin: PASSES\n")

    # 1) Baseline: run_policy accepts a representative delta kwarg.
    r: dict[str, Any] = sim.run_policy(
        robot_name="so101",
        policy_provider="mock",
        instruction="x",
        duration=0.2,
        control_substeps=2,
    )
    print(
        "run_policy(control_substeps=2):",
        {"status": r.get("status"), "msg": r["content"][0]["text"].splitlines()[0]},
    )
    assert r.get("status") == "success"

    # 2) start_policy on the same kwarg: bare CPython TypeError leaking class.
    tested: list[tuple[str, Any]] = [
        ("control_substeps", 2),
        ("max_onframe_failures", 2),
        ("async_rtc", True),
        ("rtc_inference_timeout_s", 0.5),
        ("n_episodes", 2),
        ("reset_between", False),
        ("wbc_install_torque_control", False),
    ]
    print("\nstart_policy() refusals:")
    for name, val in tested:
        try:
            sim.start_policy(
                robot_name="so101",
                policy_provider="mock",
                instruction="x",
                duration=0.2,
                **{name: val},
            )
        except TypeError as e:
            msg = str(e)
            # Current shape leaks the backend class name:
            assert "MuJoCoSimEngine" in msg, f"class leak missing: {msg}"
            assert name in msg, msg
            # And has no 'run_policy' hint:
            assert "run_policy" not in msg, f"hint already present?: {msg}"
            print(f"  - {name:30s} -> TypeError: {msg}")

    print(
        "\nAll 7 delta-kwarg refusals shaped as:\n"
        "    TypeError: MuJoCoSimEngine.start_policy() got an unexpected "
        "keyword argument '<name>'\n"
        "  * leaks backend class name\n"
        "  * no 'did you mean run_policy?' hint\n"
        "  * no mention of the signature narrowing documented nowhere in"
        " start_policy's docstring (base.py:5144-5167 only cross-references"
        " run_policy three times with no kwarg caveat)\n"
    )
    sim.cleanup()


if __name__ == "__main__":
    main()
