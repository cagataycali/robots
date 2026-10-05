"""
README quickstart papercut: the LLM-visible tool_spec description names the
robot and its joints but omits the objects and cameras the user adds — even
though both live on the same ``world`` and even though
``_world_readiness_sentence``'s docstring says the description must
"describe the session the agent is actually joining".

Reproduces the exact README quickstart (README.md L49-58) and prints the
things the live world holds vs. the things the tool description names.

Upstream:
- strands_robots/simulation/mujoco/simulation.py:6924-6983 (_world_readiness_sentence
  enumerates world.robots only; world.objects and world.cameras are read elsewhere).

For an Agent asked to "pick up the red cube", the LLM must spend one extra
tool call on list_objects/list_cameras to discover a scene the tool already
knows about. The information is cheap at description time, and the
documented purpose of this helper is to prevent exactly this round-trip
(the docstring cites "eight agent sessions on six embodiments" where a
stale opening sentence drove the first tool call into a refusal).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from strands_robots import Robot

# EXACT README quickstart
robot = Robot("so100")
robot.add_object(
    name="red_cube",
    shape="box",
    size=[0.05, 0.05, 0.05],
    position=[0.0, -0.2, 0.025],
    color=[1.0, 0.0, 0.0],
)
robot.add_camera(
    name="front",
    position=[0.3, -0.7, 0.45],
    target=[0.0, -0.2, 0.03],
)

desc = robot.tool_spec["description"]
world = robot._world

print("Live world (what the agent IS joining):")
print("  robots :", list(world.robots.keys()))
print("  objects:", list(world.objects.keys()))
print("  cameras:", list(world.cameras.keys()))

print("\nTool description head (first 320 chars):")
print(" ", desc[:320].strip())

missing = []
for name in ("red_cube", "front", "green_sphere"):
    inlined = name in desc
    print(f"\n'{name}' present in description:", inlined)
    if name in ("red_cube", "front") and not inlined:
        missing.append(name)

if missing:
    print(
        f"\nDEFECT: scene entities {missing} added through the SAME tool are not "
        f"named in the tool_spec description that the LLM reads before its first call."
    )
