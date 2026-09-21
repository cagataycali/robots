"""A panda rollout keyed by DRIVEN JOINT names: measure the motion and the report."""
import json, sys
from typing import Any
import numpy as np
import strands_robots
print("strands_robots from:", strands_robots.__file__)
from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner

KEYS = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]
TARGET = [0.0, -0.6, 0.0, -2.2, 0.0, 1.9, 0.79, 0.5]

class Ramp(Policy):
    """Walks every driven joint from its current reading to a fixed pose."""
    def __init__(self): self.n = 0
    async def get_actions(self, observation_dict, instruction, **kw):
        self.n += 1
        a = min(1.0, self.n / 100.0)
        return [{k: a * t for k, t in zip(KEYS, TARGET, strict=True)}]
    def set_robot_state_keys(self, robot_state_keys): self._k = robot_state_keys
    @property
    def requires_images(self): return False
    @property
    def provider_name(self): return "ramp"

s = Simulation(tool_name="fig", mesh=False)
s.create_world()
s.add_robot(name="arm", data_config="panda")
qpos = []
s.run_policy  # noqa
def obs(step): qpos.append([float(x) for x in s.mj_data.qpos[:8]])
res = PolicyRunner(s).run("arm", Ramp(), duration=3.0, control_frequency=50,
                          fast_mode=True, observer=obs)
pay = next(b["json"] for b in res["content"] if "json" in b)
out = {"status": res["status"],
       "action_resolution_rate": pay["action_resolution_rate"],
       "partial_action_failure_rate": pay["partial_action_failure_rate"],
       "action_errors": pay["action_errors"], "steps_used": pay["steps_used"],
       "actions_applied": pay["actions_applied"], "qpos": qpos}
print(json.dumps({k: v for k, v in out.items() if k != "qpos"}, indent=1))
q = np.array(qpos)
print("max |dq| arm joints (rad):", round(float(np.abs(q[-1][:7] - q[0][:7]).max()), 4))
json.dump(out, open(sys.argv[1], "w"))
s.cleanup()
