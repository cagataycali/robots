"""Capture a real G1 WBC rollout with a honored vs unusable height command."""
from __future__ import annotations
import glob, json, math, os, pathlib, sys
import numpy as np

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", TREE, flush=True)

from strands_robots import Robot

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--nepyope--GR00T-WholeBodyControl_g1/snapshots/*"))[0]
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DURATION, FREQ = 4.0, 50.0


def rollout(tag: str, height):
    sim = Robot("unitree_g1", mesh=False)
    sim.add_camera(name="side", position=[2.6, -2.9, 1.5], target=[0.9, 0.0, 0.7], fov=40)

    heights, xs = [], []
    # Hook the READ-ONLY observation call, not send_action: run_policy installs
    # the WBC torque shim over the actuator path, and wrapping send_action
    # displaces it (the position servos then fight the tuned PD and the G1 falls).
    real_obs = sim.get_observation

    def spy(*a, **kw):
        out = real_obs(*a, **kw)
        st = sim.get_body_state(body_name="unitree_g1/pelvis")
        js = next((c["json"] for c in st.get("content", []) if "json" in c), None)
        if js:
            heights.append(float(js["position"][2])); xs.append(float(js["position"][0]))
        return out

    sim.get_observation = spy  # type: ignore[method-assign]
    kwargs = {"target_velocity": [0.0, 0.0, 0.0]}  # standing balance
    if height is not None:
        kwargs["height"] = height

    facts = {"tag": tag, "tree": str(TREE), "height_arg": repr(height)}
    try:
        res = sim.run_policy(
            robot_name="unitree_g1", instruction="hold a standing balance", policy_provider="wbc",
            policy_config={"checkpoint": CKPT, "walk": False}, policy_kwargs=kwargs,
            duration=DURATION, control_frequency=FREQ, action_horizon=1,
        )
        facts["status"] = res.get("status", "?")
        js = next((c["json"] for c in res.get("content", []) if "json" in c), {})
        facts["n_steps"] = js.get("n_steps")
        facts["text"] = next((c["text"] for c in res.get("content", []) if "text" in c), "")[:200]
    except Exception as e:  # noqa: BLE001 - the outcome IS the measurement
        facts["status"] = "raised"
        facts["text"] = f"{type(e).__name__}: {e}"
        facts["n_steps"] = 0

    facts["ticks"] = len(heights)
    if heights:
        facts["pelvis_z_start"] = round(heights[0], 4)
        facts["pelvis_z_end"] = round(heights[-1], 4)
        facts["pelvis_z_min"] = round(min(heights), 4)
        facts["base_x_travel"] = round(xs[-1] - xs[0], 4)
        facts["n_non_finite_z"] = int(sum(not math.isfinite(h) for h in heights))
    np.save(OUT / f"{tag}_z.npy", np.asarray(heights, dtype=np.float64))
    np.save(OUT / f"{tag}_x.npy", np.asarray(xs, dtype=np.float64))

    r = sim.render(camera_name="side", width=760, height=620)
    png = next((c["image"]["source"]["bytes"] for c in r.get("content", []) if "image" in c), None)
    if png:
        (OUT / f"{tag}.png").write_bytes(png)
        facts["render"] = True
    sim.cleanup()
    print(json.dumps(facts, indent=2), flush=True)
    return facts


all_facts = {}
for tag, h in [("honored", 0.74), ("nan_height", float("nan"))]:
    all_facts[tag] = rollout(tag, h)
(OUT / "facts.json").write_text(json.dumps(all_facts, indent=2))
print("DONE")
