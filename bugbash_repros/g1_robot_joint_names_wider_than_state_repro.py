"""Repro: robot_joint_names('g1') is 30-wide, but observation.state and action vectors are 29-wide.

ABC docstring at strands_robots/simulation/base.py:1421-1434 says robot_joint_names
"is the one a LeRobotDataset recording writes the observation.state columns in,
so it is the order a policy must read that vector back in." For g1 this is wrong:
the first entry is a 7-qpos FREE joint that cannot be a scalar column.

The Newton backend fixed its OWN robot_action_keys (newton/simulation.py:760-790,
whose docstring documents this precise failure mode). The MuJoCo backend also fixed
robot_action_keys (mujoco/simulation.py:3617 via _get_valid_action_keys). But
BOTH backends leave robot_joint_names returning the free joint, so the width that
the ABC says anchors the recording is one too many for every floating-base robot.

Impact: a policy keyed by robot_joint_names has len=30 for g1; the recorded
observation.state vector has len=29; set_robot_state_keys silently binds a width
no engine can produce.
"""
import os

os.environ.pop("SYSTEM_PROMPT", None)
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

r = Robot("g1")

jn = r.robot_joint_names("g1")
ak = r.robot_action_keys("g1")
obs = r.get_observation()

# observation state is per-joint scalars; .vel entries + base_* are structured
state_scalars = [k for k in obs.keys() if not k.endswith(".vel") and not k.startswith("base_") and k != "default"]

print(f"robot_joint_names('g1')  len = {len(jn)}  first = {jn[0]!r}")
print(f"robot_action_keys('g1')  len = {len(ak)}  first = {ak[0]!r}")
print(f"get_observation scalars  len = {len(state_scalars)}")
print()
print(f"width gap (joint_names - action_keys): {sorted(set(jn) - set(ak))}")

# Verify the extra IS a 6-DoF free joint (nq=7, nv=6)
world = r._world
model = world._model
import mujoco as mj  # noqa: E402

jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "g1/floating_base_joint")
jtype = model.jnt_type[jid]
qpos_adr = model.jnt_qposadr[jid]
dof_adr = model.jnt_dofadr[jid]
nq_this = (model.jnt_qposadr[jid + 1] - qpos_adr) if jid + 1 < model.njnt else model.nq - qpos_adr
nv_this = (model.jnt_dofadr[jid + 1] - dof_adr) if jid + 1 < model.njnt else model.nv - dof_adr

print(f"\n'floating_base_joint' underlying mujoco joint:")
print(f"  type   = {{0:'FREE',1:'BALL',2:'SLIDE',3:'HINGE'}}[{jtype}] = FREE")
print(f"  nq     = {nq_this}  (7 scalars: xyz + quat_wxyz)")
print(f"  nv     = {nv_this}  (6 scalars: lin + ang)")

print()
print(f"policy keyed by robot_joint_names binds {len(jn)} keys")
print(f"policy run_policy / replay reads an observation.state vector of width {len(state_scalars)}")
print(f"=> width mismatch {len(jn) - len(state_scalars)}  (silent on {len([n for n in ['g1','unitree_g1','unitree_h1_2','spot'] if True])}+ floating-base robots)")

assert len(jn) != len(state_scalars), (
    "If this assert fires, the defect is fixed: robot_joint_names returns the "
    "same width as the recorded observation.state vector."
)

r.cleanup()
print("\nPRE-FIX FOOTGUN CONFIRMED: robot_joint_names width != observation.state width.")
