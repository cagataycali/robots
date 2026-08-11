"""Drive the production IsaacDeltaEEFController over MuJoCo seams.

The LiberoAdapter install path is duck-typed, so an engine exposing the Isaac
action seam over a real MuJoCo panda runs the REAL install decision and gives
the controller's joint targets a real physical consequence.
"""

from __future__ import annotations

import json, pathlib, sys
import numpy as np
import mujoco as mj
import strands_robots

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = pathlib.Path(TREE).name

from strands_robots.benchmarks.libero.adapter import LiberoAdapter, _ControllerInstallError
from strands_robots.simulation import Simulation

BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""
ISO_ARM = [f"panda_joint{i}" for i in range(1, 8)]
ISO_GRIP = ["panda_finger_joint1", "panda_finger_joint2"]
I2M = {f"panda_joint{i}": f"joint{i}" for i in range(1, 8)}
I2M.update({f"panda_finger_joint{i}": f"finger_joint{i}" for i in (1, 2)})
BREAK_JOINT = "panda_joint4"


class MujocoBackedIsaacSeam:
    """The five callables `_try_install_isaac_action_controller` probes."""

    def __init__(self, sim, break_observation: bool = False):
        self.sim, self.break_observation, self.installed = sim, break_observation, {}

    def list_robots(self):
        return ["panda"]

    def robot_joint_names(self, robot_name):  # the articulation DOFs
        return list(I2M)

    def get_observation(self, robot_name=None, *, skip_images=False):
        obs = self.sim.get_observation(robot_name="panda")
        return {
            iso: float(obs[mjn])
            for iso, mjn in I2M.items()
            if not (self.break_observation and iso == BREAK_JOINT)
        }

    def get_jacobian(self, body_name=None, robot_name=None, **kw):
        m, d = self.sim._world._model, self.sim._world._data
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "panda/hand")
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacBody(m, d, jp, jr, bid)
        return {"status": "success", "content": [{"text": "Jacobian"},
                {"json": {"jacp": jp.tolist(), "jacr": jr.tolist(), "nv": int(m.nv)}}]}

    def install_action_controller(self, robot_name, controller):
        self.installed[robot_name] = controller
        return {"status": "success", "content": [{"text": f"installed for {robot_name}"}]}


def hand_z(sim):
    m, d = sim._world._model, sim._world._data
    return float(d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "panda/hand")][2])


def render(sim, name):
    r = sim.render(camera_name="look", width=760, height=680)
    assert r.get("status") == "success", r
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    (OUT / f"{name}.png").write_bytes(png)
    return f"{name}.png"


def run(break_observation: bool, label: str):
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="panda", keyframe="home")
    sim.add_camera(name="look", position=[1.05, -0.95, 0.85], target=[0.42, 0.0, 0.55], fov=40)
    # add_camera recompiles the spec, which drops ctrl; without re-establishing
    # the PD targets the position servos would drive every joint to 0 and the
    # arm would drift in BOTH arms of the comparison.
    obs0 = sim.get_observation(robot_name="panda")
    hold = {mjn: float(obs0[mjn]) for mjn in I2M.values()}
    sim.send_action(hold, robot_name="panda", n_substeps=10)
    sim.step(80)
    rec = {"label": label, "z_start": hand_z(sim), "applied": 0, "action_errors": 0}
    rec["png_start"] = render(sim, f"{label}_start")

    adapter = LiberoAdapter.from_text(BDDL, strict_action_controller=True)
    seam = MujocoBackedIsaacSeam(sim, break_observation=break_observation)
    try:
        adapter._install_action_controller(seam)
        rec["install"] = "accepted"
        rec["install_error"] = None
    except _ControllerInstallError as e:
        rec["install"] = "refused"
        rec["install_error"] = str(e).split(" GR00T actions")[0]
    rec["adapter_error"] = adapter._action_controller_error
    rec["controller_installed"] = bool(seam.installed)

    ctrl = seam.installed.get("panda")
    if ctrl is not None:
        for _ in range(26):
            # Mirror IsaacSimulation.send_action: convert, then apply. A
            # conversion failure is an error envelope, never a fall-through.
            try:
                targets = ctrl.compute_joint_targets({"z": -1.0, "gripper": 1.0})
            except (RuntimeError, ValueError, TypeError):
                rec["action_errors"] += 1
                sim.send_action(hold, robot_name="panda", n_substeps=10)
                continue
            mapped = {I2M[k]: v for k, v in targets.items()}
            res = sim.send_action(mapped, robot_name="panda", n_substeps=10)
            if res.get("status") == "success":
                rec["applied"] += 1
            else:
                rec["action_errors"] += 1
    else:
        # Isaac applies nothing on an error envelope, so the articulation keeps
        # its last commanded targets. Re-sending the hold models exactly that.
        for _ in range(26):
            sim.send_action(hold, robot_name="panda", n_substeps=10)

    rec["z_end"] = hand_z(sim)
    rec["png_end"] = render(sim, f"{label}_end")
    sim.cleanup()
    return rec


facts = {"tree": TREE, "runs": [run(False, "healthy"), run(True, "broken_obs")]}
(OUT / f"facts_{TAG}.json").write_text(json.dumps(facts, indent=2))
for r in facts["runs"]:
    print(f"  {r['label']:<11} install={r['install']:<8} adapter_error={'set' if r['adapter_error'] else 'None':<4} "
          f"applied={r['applied']:>2}/26 errs={r['action_errors']:>2} z {r['z_start']:.4f} -> {r['z_end']:.4f}")
