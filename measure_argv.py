"""Measure what build_lerobot_command actually puts on the lerobot argv."""
from __future__ import annotations
import json, pathlib, sys
from typing import Any
import strands_robots.tools.lerobot_teleoperate as m

TREE = str(pathlib.Path(m.__file__).parents[2])
print("TREE:", TREE, file=sys.stderr)
B = m.build_lerobot_command

REC = dict(action="start", robot_type="so101_follower", robot_port="/dev/ttyACM1",
           teleop_type="so101_leader", teleop_port="/dev/ttyACM0",
           dataset_repo_id="user/pick", dataset_single_task="pick the cube")
TEL = dict(action="start", robot_type="so101_follower", robot_port="/dev/ttyACM1",
           teleop_type="so101_leader", teleop_port="/dev/ttyACM0")
RPL = dict(action="replay", robot_type="so101_follower", robot_port="/dev/ttyACM1",
           dataset_repo_id="user/pick")

def call(base: dict[str, Any], flag: str, **kw: Any) -> dict[str, Any]:
    args = dict(base); args.update(kw)
    try:
        argv = B(**args)
    except Exception as e:
        return {"outcome": "refused", "detail": str(e), "argv": None, "token": None}
    tok = argv[argv.index(flag) + 1] if flag in argv else None
    return {"outcome": "built", "detail": None, "argv": argv,
            "token": tok, "flag_present": flag in argv}

rows: list[dict[str, Any]] = []
NAN, INF = float("nan"), float("inf")
for label, val in [("0", 0), ("-5", -5), ("2.7", 2.7), ("nan", NAN), ("inf", INF),
                   ("True", True), ("'30'", "30"), ("None", None), ("[30]", [30])]:
    r = call(REC, "--dataset.fps", dataset_fps=val)
    rows.append({"knob": "dataset_fps", "mode": "record", "value": label, **r})
for label, val in [("0", 0), ("-1", -1), ("nan", NAN)]:
    r = call(REC, "--dataset.num_episodes", dataset_num_episodes=val)
    rows.append({"knob": "dataset_num_episodes", "mode": "record", "value": label, **r})
rows.append({"knob": "dataset_episode_time_s", "mode": "record", "value": "0",
             **call(REC, "--dataset.episode_time_s", dataset_episode_time_s=0)})
for label, val in [("-1", -1), ("nan", NAN)]:
    rows.append({"knob": "replay_episode", "mode": "replay", "value": label,
                 **call(RPL, "--dataset.episode", replay_episode=val)})

# The two truthiness reads, where 0 inverted the request.
falsy = {
    "teleop_time_s=0": call(TEL, "--teleop_time_s", teleop_time_s=0),
    "replay dataset_fps=0": call(RPL, "--dataset.fps", dataset_fps=0),
}

# Honored calls: must be byte-identical across trees.
honored = {
    "record fps=30": call(REC, "--dataset.fps", dataset_fps=30),
    "record fps=30.0": call(REC, "--dataset.fps", dataset_fps=30.0),
    "record reset_time_s=0": call(REC, "--dataset.reset_time_s", dataset_reset_time_s=0),
    "replay episode=0": call(RPL, "--dataset.episode", replay_episode=0),
    "teleop budget=12.5": call(TEL, "--teleop_time_s", teleop_time_s=12.5),
    "teleop budget=None": call(TEL, "--teleop_time_s", teleop_time_s=None),
}
json.dump({"tree": TREE, "rows": rows, "falsy": falsy, "honored": honored},
          open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], file=sys.stderr)
