"""
Repro: `move_to` tool_spec.json advertises `default 0.01` for `tol`, but the
Python function (both MuJoCo + Isaac) actually uses `tol=0.015`. The fix for
the Python side merged via harness #487 (SIM-BATCH-1b); the JSON schema the
LLM reads at tool-registration time was never updated to match.

Class: Docs mismatch (schema-vs-code) + Error UX
Target: README quickstart — the hero example routes `pick up the red cube`
through `move_to` + `set_gripper`, and an Agent plans `tol` from the
tool_spec it sees.

Expected: schema `default` matches the Python signature; so an LLM that
drops to the next-tighter tol reasons about the same number the backend
converges on.
Actual:
    * strands_robots/simulation/mujoco/tool_spec.json:389 says "default 0.01"
    * strands_robots/simulation/mujoco/motion_primitives.py:478  tol=0.015
    * strands_robots/simulation/isaac/motion_primitives.py:722    tol=0.015
    * strands_robots/simulation/mujoco/simulation.py:4055  describe says tol=0.015
    * strands_robots/simulation/mujoco/motion_primitives.py:541  docstring says
      "default 0.015 is where the so100 position servos settle within"

An LLM that reads the schema sees `tol=0.01` as the baseline and will often
retry a reachable so100 target believing that bar is the one the backend
tests against — exactly the failure mode #487 fixed on the code side, now
leaking back through the schema.

Fix: schema side only — flip tool_spec.json:389 from
`default 0.01` to `default 0.015` so JSON schema and Python agree.

Run:  python bugbash_repros/move_to_tol_schema_vs_code_mismatch_repro.py
"""
import inspect, json, re, sys
from strands_robots import Robot

r = Robot("so100")

# 1) What the LLM sees (JSON schema shipped with the tool)
tol_desc = r.tool_spec["inputSchema"]["json"]["properties"]["tol"]["description"]
m = re.search(r"meters for move_to.*?default\s+([0-9.]+)", tol_desc)
schema_default = m.group(1) if m else "NOT FOUND"

# 2) What the function actually defaults to (merged in harness #487)
from strands_robots.simulation.mujoco.motion_primitives import MotionPrimitivesMixin
code_default = inspect.signature(MotionPrimitivesMixin.move_to).parameters["tol"].default

# 3) Isaac must agree with mujoco; it does
from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin
isaac_default = inspect.signature(IsaacMotionPrimitivesMixin.move_to).parameters["tol"].default

print(json.dumps({
    "schema_tol_default_the_llm_reads": schema_default,
    "python_tol_default_mujoco": code_default,
    "python_tol_default_isaac": isaac_default,
    "mismatch": str(schema_default) != str(code_default),
}, indent=2))

assert str(schema_default) == str(code_default), (
    f"FAIL: tool_spec.json advertises `default {schema_default}` but the "
    f"Python signature uses `tol={code_default}`. An LLM that plans a "
    f"move_to retry from the schema will overshoot the real convergence "
    f"gate; harness #487 fixed the Python side but the JSON schema was "
    f"never updated."
)
print("PASS: schema and code agree on move_to tol default")
