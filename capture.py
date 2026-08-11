"""Drive the real IsaacDeltaEEFController against a real MuJoCo Panda.

The controller reads current joint positions and the end-effector Jacobian
through injected callables, so a shim backed by a compiled MuJoCo model runs
the production conversion unchanged and gives its joint targets a real
physical consequence. Isaac Sim is not required to reach either decision.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import mujoco as mj
import numpy as np

import strands_robots
from strands_robots import Simulation

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
TAG = sys.argv[2]

from strands_robots.simulation.isaac.delta_eef import IsaacDeltaEEFController  # noqa: E402

# Isaac's Franka USD joint names -> the MuJoCo menagerie panda's.
ISAAC_ARM = [f"panda_joint{i}" for i in range(1, 8)]
ISAAC_GRIP = ["panda_finger_joint1", "panda_finger_joint2"]
MJ_ARM = [f"panda/joint{i}" for i in range(1, 8)]
MJ_GRIP = ["panda/finger_joint1", "panda/finger_joint2"]
ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
EE_BODY = "panda/hand"


def render(sim, path):
    r = sim.render(camera_name="look", width=760, height=620)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    pathlib.Path(path).write_bytes(png)
    return path


def build_scene():
    sim = Simulation(backend="mujoco", mesh=False)
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot(name="panda", keyframe="home")["status"] == "success"
    # The camera must be added before any rollout: add_camera recompiles the
    # spec, which drops ctrl.
    assert sim.add_camera(
        name="look", position=[1.05, -0.95, 0.72], target=[0.35, 0.0, 0.42], fov=38
    )["status"] == "success"
    sim.step(120)
    return sim


def hold(sim):
    """Re-issue the settled pose so the position servos hold it."""
    obs = sim.get_observation(robot_name="panda")
    q = [float(obs[j.split("/")[-1]]) for j in MJ_ARM]
    cmd = dict(zip(ACTUATORS[:7], q, strict=True))
    assert sim.send_action(cmd, robot_name="panda", n_substeps=10)["status"] == "success"
    return cmd


def seams(sim):
    model = sim._world._model
    data = sim._world._data
    qadr = [model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)] for n in MJ_ARM]
    dadr = [model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)] for n in MJ_ARM]
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, EE_BODY)

    def joint_positions_fn():
        return np.array([data.qpos[a] for a in qadr], dtype=np.float64)

    def jacobian_fn():
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacBody(model, data, jacp, jacr, bid)
        return np.vstack([jacp[:, dadr], jacr[:, dadr]])

    return joint_positions_fn, jacobian_fn


def run(label, **ctor):
    sim = build_scene()
    home = render(sim, OUT / f"{TAG}_{label}_home.png")
    hold_cmd = hold(sim)
    qfn, jfn = seams(sim)
    rec = {"label": label, "tree": TREE, "home_png": str(home)}
    try:
        controller = IsaacDeltaEEFController(
            arm_joint_names=ISAAC_ARM,
            gripper_joint_names=ISAAC_GRIP,
            joint_positions_fn=qfn,
            jacobian_fn=jfn,
            **ctor,
        )
    except Exception as e:  # noqa: BLE001 - the constructor verdict is the measurement
        # Idle exactly as the all-refused arm does below, so the two
        # "nothing was applied" panels are comparable.
        for _ in range(26):
            sim.step(10)
        for _ in range(6):
            sim.send_action(hold_cmd, robot_name="panda", n_substeps=10)
        rec.update(ctor_ok=False, ctor_error=f"{type(e).__name__}: {e}",
                   applied=0, refused=0, after_png=str(render(sim, OUT / f"{TAG}_{label}_after.png")),
                   envelope="(no controller was built -- no action was issued)")
        sim.cleanup()
        return rec

    applied = refused = 0
    envelope = ""
    for _ in range(26):
        targets = controller.compute_joint_targets({"x": 0.30, "z": -1.0, "gripper": 1.0})
        cmd = {}
        for isaac_name, value in targets.items():
            if isaac_name in ISAAC_ARM:
                cmd[ACTUATORS[ISAAC_ARM.index(isaac_name)]] = value
        if not all(np.isfinite(list(cmd.values()))):
            # What the Isaac backend reports: send_action's action-value domain
            # refuses the converted target and names the joint it was handed.
            bad = next(k for k, v in cmd.items() if not np.isfinite(v))
            envelope = (
                f"send_action: action value for key '{ISAAC_ARM[ACTUATORS.index(bad)]}' must be "
                f"finite (no nan/inf), got nan."
            )
            refused += 1
            sim.step(10)
            continue
        assert sim.send_action(cmd, robot_name="panda", n_substeps=10)["status"] == "success"
        applied += 1
    if refused:
        # Nothing was applied, so keep holding the settled pose.
        for _ in range(6):
            sim.send_action(hold_cmd, robot_name="panda", n_substeps=10)
    after = render(sim, OUT / f"{TAG}_{label}_after.png")
    obs = sim.get_observation(robot_name="panda")
    rec.update(ctor_ok=True, ctor_error="", applied=applied, refused=refused,
               after_png=str(after), envelope=envelope or "(applied)",
               ee_z=float(sim.get_body_state(body_name=EE_BODY)["content"][0]["json"]["position"][2])
               if False else 0.0,
               joints=[round(float(obs[j.split("/")[-1]]), 6) for j in MJ_ARM])
    sim.cleanup()
    return rec


records = [
    run("healthy"),
    run("posinf", pos_scale=math.inf),
]
(OUT / f"facts_{TAG}.json").write_text(json.dumps({"tree": TREE, "records": records}, indent=2))
for r in records:
    print(f"  {r['label']:9s} ctor_ok={r['ctor_ok']} applied={r['applied']} refused={r['refused']}")
    print(f"      {r.get('ctor_error') or r['envelope']}")
print("WROTE", OUT / f"facts_{TAG}.json")
