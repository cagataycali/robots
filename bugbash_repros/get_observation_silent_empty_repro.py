"""
Minimal repro for strands-labs/robots v0.5.3 defect:
MuJoCoSimEngine.get_observation() returns {} silently on 3 degraded-mode paths
(torn-down world, zero-robot world, unknown robot_name) while:
- 24 sibling methods in the same file return status=error + _NO_WORLD_MSG
- The Isaac backend logs WARNING + remedy before returning {}
- The ABC schema docstring in base.py:1875-1942 makes NO mention of {} as
  a documented degraded return

Impact:
  A rollout / evaluation loop that reads obs = robot.get_observation() and
  branches on obs truthiness (common pattern) iterates on an empty dict with
  no indication the sim is dead - all downstream obs["joint_name"] accesses
  KeyError on the first step while the surface reported success.

Run:
  MUJOCO_GL=egl python get_observation_silent_empty_repro.py

Expected: either status=error (sibling convention), OR a logger.warning
          naming the degraded mode (Isaac convention).
Actual:   bare {} with no log, no status.
"""
import logging, os
os.environ.setdefault("MUJOCO_GL", "egl")

# Capture any log records emitted by the backend
import io
log_buf = io.StringIO()
logging.basicConfig(level=logging.DEBUG, stream=log_buf, force=True)

from strands_robots import Robot

r = Robot("so101")

# Case A: Torn-down world
assert r.get_observation(), "pre-destroy obs should be non-empty"
res = r.destroy()
assert res["status"] == "success", f"destroy failed: {res}"

obs_torn = r.get_observation()
send_torn = r.send_action({"shoulder_pan": 0.1})
step_torn = r.step()
state_torn = r.get_robot_state("so101")

print(f"after destroy -> get_observation():  {obs_torn!r}  (silent empty)")
print(f"after destroy -> send_action():      status={send_torn['status']!r}  "
      f"text={send_torn['content'][0]['text']!r}")
print(f"after destroy -> step():             status={step_torn['status']!r}  "
      f"text={step_torn['content'][0]['text']!r}")
print(f"after destroy -> get_robot_state():  status={state_torn['status']!r}  "
      f"text={state_torn['content'][0]['text']!r}")

# Case B: never-created world
r2 = Robot("so101")
r2.destroy()  # reuse instance, now worldless
obs_dead = r2.get_observation()
print(f"\nworldless  -> get_observation():  {obs_dead!r}  (silent empty)")

# Compare to Isaac convention: scan captured logs for any WARNING about get_observation
log_output = log_buf.getvalue()
has_log = any("get_observation" in line and ("WARNING" in line or "warning" in line)
              for line in log_output.splitlines())
print(f"\nMuJoCo backend logged a WARNING on silent empty? {has_log}")
print("(Isaac backend logs 5 named conditions in simulation/isaac/simulation.py:5639-5667;"
      " MuJoCo backend has 3 bare `return {}` at simulation/mujoco/simulation.py:1232,1234,1237"
      " with zero logging.)")
