"""Repro: README quickstart second-run shows the shortest error of any sibling.

Reproduce from a fresh checkout:

    MUJOCO_GL=egl python bugbash_repros/add_object_dup_truncated_vs_siblings_repro.py

Pre-fix output (mujoco/simulation.py:4984):

    add_object (second call): "Object 'red_cube' exists."
    add_camera (second call): "add_camera: camera 'front' already exists. Remove it first."
    add_robot  (second call): "Robot 'so100' already exists. Pick a different name, or omit name= to auto-number. Existing: so100."

Note:
 - missing the word "already"            (every sibling in the same class has it)
 - missing the "add_object:" verb prefix (every sibling scene-mutator has it)
 - missing a remedy                      (add_camera / add_robot both carry one)
 - inconsistent with newton + isaac, which both say "Object '{name}' already exists."
 - inconsistent with the test comment at
   tests/simulation/test_object_mass_domain_across_backends.py:249 which quotes
   "Object 'crate' already exists." as the expected message.

Context (why README-quickstart surface):
 - README's three-line snippet (post harness#724) teaches
   ``robot.add_object(name="red_cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.0, -0.2, 0.025])``
   as the first scene-mutator a user calls. A user re-running the snippet (or
   iterating in a notebook / REPL loop) hits this message immediately with no
   recovery hint while two sibling methods on the same object carry one.
"""

import os
import warnings

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

from strands_robots import Robot

robot = Robot("so100")

# --- README-quickstart second call: add_object dup -------------------------
robot.add_object(
    name="red_cube",
    shape="box",
    size=[0.05, 0.05, 0.05],
    position=[0.0, -0.2, 0.025],
    color=[1.0, 0.0, 0.0],
)
dup_object = robot.add_object(
    name="red_cube",
    shape="box",
    size=[0.05, 0.05, 0.05],
    position=[0.0, -0.2, 0.025],
    color=[1.0, 0.0, 0.0],
)
print("add_object (second call):", repr(dup_object["content"][0]["text"]))

# --- Sibling 1: add_camera dup (same class, same file) --------------------
robot.add_camera(name="front", position=[0.3, -0.7, 0.45], target=[0.0, -0.2, 0.03])
dup_camera = robot.add_camera(
    name="front", position=[0.3, -0.7, 0.45], target=[0.0, -0.2, 0.03]
)
print("add_camera (second call):", repr(dup_camera["content"][0]["text"]))

# --- Sibling 2: add_robot dup (same class, same file) ---------------------
dup_robot = robot.add_robot(name="so100")
print("add_robot  (second call):", repr(dup_robot["content"][0]["text"]))

# --- Assertion ------------------------------------------------------------
# After the fix the three refusals share the shape:
#   "<verb>: <noun> '{name}' already exists. <remedy>."
# Pre-fix, add_object was the only sibling missing "already" + verb prefix +
# remedy.
text = dup_object["content"][0]["text"]
assert "already exists" in text, (
    "add_object dup refusal is still truncated (missing 'already exists')"
)
assert text.startswith("add_object:"), (
    "add_object dup refusal is still missing the 'add_object:' verb prefix"
)
assert "remove_object" in text.lower() or "remove it first" in text.lower(), (
    "add_object dup refusal is still missing a remedy (add_camera has 'Remove it first.')"
)
print("\nOK: add_object dup refusal now has parity with add_camera / add_robot.")
