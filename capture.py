"""Capture what each posture-flag spelling puts on the lerobot argv."""

import json
import pathlib
import sys

import strands_robots.tools.lerobot_teleoperate as m

TREE = str(pathlib.Path(m.__file__).parents[2])

RECORD = dict(
    action="start", robot_type="so101_follower", robot_port="/dev/ttyACM0",
    teleop_type="so101_leader", teleop_port="/dev/ttyACM1",
    dataset_repo_id="user/ds", dataset_single_task="pick the cube",
)
TOKEN = {
    "dataset_push_to_hub": "--dataset.push_to_hub",
    "dataset_video": "--dataset.video",
    "display_data": "--display_data",
    "dagger_record_autonomous": "--strategy.record_autonomous",
}
OPT_OUTS = ["false", "no", "off", "0"]


def emitted(flag, value):
    try:
        argv = m.build_lerobot_command(**dict(RECORD, **{flag: value}))
    except ValueError as exc:
        return {"outcome": "refused", "detail": str(exc).split(". It selects")[0]}
    tok = TOKEN[flag]
    if tok not in argv:
        return {"outcome": "omitted", "detail": f"{tok} absent"}
    nxt = argv.index(tok) + 1
    val = argv[nxt] if nxt < len(argv) and not argv[nxt].startswith("--") else "(bare)"
    return {"outcome": "emitted", "detail": f"{tok} {val}"}


rows = {}
for flag in ("dataset_push_to_hub", "dataset_video", "display_data"):
    rows[flag] = {v: emitted(flag, v) for v in OPT_OUTS}

# no-regression: the honored argv for a real opt-out / opt-in
honored = {
    "opt_out_false": m.build_lerobot_command(**dict(RECORD, dataset_push_to_hub=False)),
    "opt_in_true": m.build_lerobot_command(**dict(RECORD, dataset_push_to_hub=True)),
}

# the agent-facing schema description for the two undocumented flags
props = m.lerobot_teleoperate.tool_spec["inputSchema"]["json"]["properties"]
schema = {
    k: " ".join(props[k]["description"].split())[:78]
    for k in ("auto_accept_calibration", "dagger_record_autonomous")
}

out = {"tree": TREE, "rows": rows, "honored": honored, "schema": schema}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print("TREE:", TREE)
for flag, per in rows.items():
    print(f"  {flag}: " + " | ".join(f"{v}->{r['outcome']}" for v, r in per.items()))
