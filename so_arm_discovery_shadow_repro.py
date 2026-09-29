#!/usr/bin/env python
"""
Repro: Robot("so_arm101") silently loads a *degraded* SO-101 without joint labels.

CONTEXT
-------
`list_discoverable()` (`strands_robots.registry.discovery.list_discoverable`)
lists names harvested from `robot_descriptions`. Its docstring explains a
curated `robots.json` entry always wins over discovery. That is only true when
the user types the *canonical* name. Five description short-names ─ `n1`,
`openarm_v1`, `so_arm101`, `viper`, `widow` ─ point at the *same MJCF model*
as an already-curated robot (`fourier_n1`, `openarm`, `so101`, `vx300s`,
`wx250s`), but the registry does NOT list them as aliases. So `Robot(short)`
falls through to the discovery path and returns a robot with:

  - `joint_labels` empty (curated entry has `{"1": "shoulder_pan", ...}`)
  - the `send_action({"shoulder_pan": ...})` path documented in the quickstart
    returns `status=error` instead of moving the arm
  - `hardware`, `gripper`, `aliases` metadata dropped

The user gets no warning. Both `Robot("so101")` and `Robot("so_arm101")` load
successfully; only the labelled action call ─ the one first-robot.md teaches ─
reveals the divergence.

REPRO
-----
    python so_arm_discovery_shadow_repro.py

EXPECTED (following docs/start/first-robot.md)
----------------------------------------------
    Robot('so_arm101') behaves like Robot('so101'):
    - joint_labels present
    - send_action({"shoulder_pan": 0.3}) → status=success

ACTUAL
------
    Robot('so_arm101') is degraded:
    - joint_labels: {}
    - send_action({"shoulder_pan": 0.3}) → status=error

UPSTREAM
--------
strands_robots/registry/discovery.py list_discoverable() yields 57 names
that duplicate a curated model. 5 of those names have no alias entry in the
curated `robots.json`, so the discovery path resolves them instead of the
curated one.
"""
from strands_robots import Robot
from strands_robots.registry.robots import get_robot, resolve_name
from strands_robots.registry.discovery import list_discoverable


def probe(name: str) -> dict:
    r = Robot(name, mesh=False)
    try:
        state = r.get_robot_state()["content"][1]["json"]
        labels = state.get("joint_labels", {})
        action_res = r.send_action({"shoulder_pan": 0.3}, n_substeps=10)
        return {
            "in_world": r.list_robots()[0],
            "labels": labels,
            "labelled_send_action_status": action_res.get("status"),
            "labelled_send_action_text": action_res["content"][0].get("text"),
            "registered": get_robot(name) is not None,
            "resolves_to": resolve_name(name),
            "in_list_discoverable": name in list_discoverable(),
        }
    finally:
        r.cleanup()


if __name__ == "__main__":
    for name in ("so101", "so_arm101"):
        print(f"\n=== Robot({name!r}) ===")
        result = probe(name)
        for k, v in result.items():
            print(f"  {k}: {v!r}")

    print("\n-- Assertions --")
    canonical = probe("so101")
    shadow = probe("so_arm101")
    assert canonical["labels"], "so101 should have joint labels"
    assert canonical["labelled_send_action_status"] == "success"
    # These are the failing assertions:
    assert shadow["labels"], (
        f"so_arm101 has empty joint_labels; user cannot use 'shoulder_pan' etc. "
        f"Got: {shadow["labels"]!r}"
    )
    assert shadow["labelled_send_action_status"] == "success", (
        f"so_arm101 refuses labelled send_action even though same physical arm. "
        f"Got: status={shadow["labelled_send_action_status"]!r}, "
        f"text={shadow["labelled_send_action_text"]!r}"
    )
    print("OK")
