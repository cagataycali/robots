"""get_features() hides joint_labels for so101 — discovery call, no record promise.

The so101 registry entry carries::

    "joint_labels": {"1": "shoulder_pan", "2": "shoulder_lift",
                     "3": "elbow_flex", "4": "wrist_flex",
                     "5": "wrist_roll", "6": "gripper"}
(``strands_robots/registry/robots.json``:293-301)

Two sibling calls on the SAME object expose the label map:

* ``get_robot_state`` -> state.json ships ``joint_labels`` and the text block reads
  ``1 (shoulder_pan): pos=…``;
* ``send_action({'Rotation': 0.3})`` -> refusal names the label mapping
  verbatim.

get_features -- the method whose own docstring on
``strands_robots/simulation/base.py`` is "describe the simulation's joints /
actuators / cameras / robots" and whose MuJoCo override docstring sentence 1 is
"Describe the simulation's joints / actuators / cameras / robots" -- never
calls ``_robot_joint_labels``. Its ``json.features.robots.{name}`` subdict
reports ``{joint_names, n_joints, n_actuators, data_config, source}`` and
nothing semantic.

This is NOT the same closed class as:

* harness#712 (``get_observation`` numeric for so101) -- that was closed R-A18
  because the schema's dataset-column stability promise forbids adding alias
  keys to the observation; ``get_robot_state`` was given a sidecar instead.
  ``get_features`` carries no column promise and already has a sidecar.
* harness#768 (``robot_action_keys`` returns actuator names on lekiwi) --
  closed with the same recording-stability reasoning (``base.py``:2275 and
  every backend's recorder keys off this list), and shipped a per-robot-page
  docs fix. ``get_features.robots.{name}`` is not consumed by any recorder.
* harness#714 (``robot_joint_names`` returned ``[]`` on unknown name) --
  silent-wrong on name resolution, orthogonal.

What this repro proves:

1. ``get_features`` already has a per-robot sidecar in ``.content[1].json.features.robots.{name}``.
2. That sidecar's shape is pure discovery metadata (``data_config``, ``source``,
   joint/actuator counts, joint names) -- it never flows into a recorder or
   policy-key binding.
3. The ``joint_labels`` map the registry carries, that ``get_robot_state``
   already renders, is NOT in that sidecar.
4. A fresh user who added ``get_features`` to their intro-script "what does this
   robot have?" page learns the arm has six joints named 1..6 and no label hint
   -- the same experience the ``_world_readiness_sentence`` fix (harness#758)
   closed on the tool_spec hot path. ``get_features`` is the next hop a reader
   takes, and it still drops the labels.

Run (fresh subprocess, no shared sim state)::

    MUJOCO_GL=egl python bugbash_repros/so101_get_features_hides_joint_labels_repro.py

Expected: ``get_features(robot_name='so101').content[1].json.features.robots.so101``
  carries a ``joint_labels`` key with the registry's label dict.
Actual:   that key is absent; the sidecar is numeric-name-only.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.registry.robots import joint_labels


def _features_sidecar(robot: Robot, name: str) -> dict:
    f = robot.get_features(robot_name=name)
    for item in f.get("content", []):
        if "json" in item:
            return item["json"]["features"]["robots"][name]
    raise AssertionError("get_features has no JSON sidecar")


def main() -> None:
    print(f"strands_robots joint_labels('so101') (registry truth):")
    print(f"  {joint_labels('so101')}\n")

    r = Robot("so101")

    # 1) get_features sidecar: our focus.
    side = _features_sidecar(r, "so101")
    print("get_features(robot_name='so101').json.features.robots.so101 (keys):")
    print(f"  {list(side.keys())}")
    print(f"  joint_names: {side['joint_names']}")
    has_labels = "joint_labels" in side
    print(f"  joint_labels: {'MISSING' if not has_labels else side['joint_labels']}\n")

    # 2) sibling calls already surface it -> consistency gap.
    rs = r.get_robot_state("so101")
    rs_sidecar = None
    for item in rs.get("content", []):
        if "json" in item:
            rs_sidecar = item["json"]
    print("get_robot_state (sibling discovery call -- ALREADY carries labels):")
    print(f"  json.keys: {list(rs_sidecar.keys()) if rs_sidecar else '?'}")
    if rs_sidecar and "joint_labels" in rs_sidecar:
        print(f"  joint_labels: {rs_sidecar['joint_labels']}")
    print()

    # 3) text block agent-first (same reasoning as the hot path).
    text = next(
        (item["text"] for item in r.get_features(robot_name="so101").get("content", []) if "text" in item),
        "",
    )
    print(f"get_features .text (first 300 chars):\n  {text[:300]}\n")

    if has_labels:
        print(
            "✓ get_features now carries joint_labels for so101 (post-fix / applied patch).\n"
            "  Legacy robots without a registry label entry still get {} -- no shape\n"
            "  regression for panda / g1 / go2 / aloha."
        )
    else:
        print(
            "✗ get_features's per-robot sidecar has no joint_labels. Agent that chose\n"
            "  get_features as its discovery call learns the arm has six joints named\n"
            "  1..6 and no semantic hint; sibling get_robot_state would have told it\n"
            "  '1 (shoulder_pan)..6 (gripper)'."
        )


if __name__ == "__main__":
    main()
