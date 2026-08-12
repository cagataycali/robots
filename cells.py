"""Cell table: which teardown paths release the action-controller registration?"""
import json, pathlib, sys
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
import numpy as np, mujoco
from typing import cast
from strands_robots.policies.wbc import WBCConfig, WBCPolicy, install_wbc_torque_control
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.model_registry import resolve_model

class _S:
    class _I: name = "obs"
    def get_inputs(self): return [self._I()]
    def run(self, o, f): return [np.zeros((1, 15), dtype=np.float32)]

def pol():
    p = WBCPolicy(config=WBCConfig(policy_path="x.onnx"), walk=False, allow_missing_models=True)
    p.policy_session = _S(); return p

class _R:
    def __init__(s, ns): s.namespace = ns
class _W:
    def __init__(s, m, d, ns):
        s._model, s._data = m, d
        s.robots = {"unitree_g1": _R(ns)}
        s._backend_state = {}
class _Sim:
    def __init__(s, w): s._world = w

def fake():
    m = mujoco.MjModel.from_xml_path(resolve_model("unitree_g1"))
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, 1) or ""
    ns = "unitree_g1/" if n.startswith("unitree_g1/") else ""
    return _Sim(_W(m, mujoco.MjData(m), ns))

cells = {}

# CELL 1 - the DOCUMENTED MANUAL path: install + controller.uninstall()
sim = fake()
c = install_wbc_torque_control(cast(SimEngine, sim), pol(), "unitree_g1")
c.uninstall()
cells["manual_install_then_uninstall_releases"] = "action_controller" not in sim._world._backend_state

# CELL 2 - the AUTO path: hook install + its returned cleanup
from strands_robots.simulation.mujoco.simulation import Simulation
sim2 = fake(); eng = Simulation(); eng._world = sim2._world
cleanup = eng._maybe_install_wbc_torque_control(pol(), "unitree_g1")
assert callable(cleanup)
cleanup()
cells["auto_hook_cleanup_releases"] = "action_controller" not in sim2._world._backend_state

# CELL 3 - identity guard on the MANUAL path
sim3 = fake()
c3 = install_wbc_torque_control(cast(SimEngine, sim3), pol(), "unitree_g1")
newer = object(); sim3._world._backend_state["action_controller"] = newer
c3.uninstall()
cells["manual_uninstall_spares_a_newer_controller"] = sim3._world._backend_state.get("action_controller") is newer

# CELL 4 - identity guard on the AUTO path
sim4 = fake(); eng4 = Simulation(); eng4._world = sim4._world
cl4 = eng4._maybe_install_wbc_torque_control(pol(), "unitree_g1")
newer4 = object(); sim4._world._backend_state["action_controller"] = newer4
cl4()
cells["auto_cleanup_spares_a_newer_controller"] = sim4._world._backend_state.get("action_controller") is newer4

# CELL 5 - AUTO path when uninstall() raises partway: is the registry still released?
sim5 = fake(); eng5 = Simulation(); eng5._world = sim5._world
cl5 = eng5._maybe_install_wbc_torque_control(pol(), "unitree_g1")
ctrl5 = sim5._world._backend_state["action_controller"]
type(ctrl5).uninstall_orig = type(ctrl5).uninstall
def boom(self): raise RuntimeError("gain restore failed")
type(ctrl5).uninstall = boom
try:
    cl5()
except RuntimeError:
    pass
type(ctrl5).uninstall = type(ctrl5).uninstall_orig
cells["auto_releases_registry_even_if_uninstall_raises"] = "action_controller" not in sim5._world._backend_state

print(json.dumps({"tree": TREE, "cells": cells}, indent=2))
json.dump({"tree": TREE, "cells": cells}, open(sys.argv[1], "w"), indent=2)
