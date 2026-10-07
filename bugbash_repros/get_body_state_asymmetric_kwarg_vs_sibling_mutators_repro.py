"""
Repro: README-quickstart close-on-nothing advice "get_body_state gives its position"
loops the user because get_body_state uses body_name= while every sibling
scene-mutator (add_object, remove_object, move_object, add_camera, remove_camera,
add_robot, remove_robot) uses name=.

Direct Python call -> stock TypeError, no "Did you mean" hint.
Agent-dispatch path (sim(action=..., name=...)) HAS the hint; direct call does not.

Expected: get_body_state(name="red_cube") either accepts name= as alias or
          emits a "did you mean body_name?" hint, same as the agent path.
Actual:   TypeError: PhysicsMixin.get_body_state() got an unexpected keyword
          argument "name"

Upstream:
  - strands_robots/simulation/mujoco/physics.py:1694 (get_body_state signature)
  - strands_robots/simulation/motion_primitives_base.py:926-929 (set_gripper
    close-on-nothing advice recommending get_body_state)
  - strands_robots/simulation/mujoco/simulation.py:_unknown_param dispatch
    (agent path HAS did-you-mean hint; direct method call does not)
"""
from strands_robots import Robot

robot = Robot("so100")
robot.add_object(name="red_cube", shape="box", size=[0.05, 0.05, 0.05],
                 position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0])

# User follows the README exactly: fingers close on nothing -> text says
# "move_to the object first (get_body_state gives its position)".
r_close = robot.set_gripper(robot_name="so100", state="close")
advice = r_close["content"][0]["text"]
assert "get_body_state gives its position" in advice, advice

# What a user types next, consistent with every sibling verb that uses name=:
try:
    robot.get_body_state(name="red_cube")   # <-- the natural call
    print("UNEXPECTED: no error")
except TypeError as e:
    assert "unexpected keyword argument" in str(e), e
    print(f"[Direct call]  TypeError (no hint): {e}")

# The agent-dispatch path HAS the did-you-mean hint, which proves the fix is
# already written next door:
res = robot(action="get_body_state", name="red_cube")
print(f"[Agent path]   status={res['status']}")
print(f"[Agent path]   {res['content'][0]['text'][:220]}")

# The correct kwarg works fine
ok = robot.get_body_state(body_name="red_cube")
assert ok["status"] == "success", ok
print(f"[Correct call] {ok['content'][0]['text'].splitlines()[0]}")

# Sibling drift table (reproducible via inspect.signature):
#   add_object       (name, ...)
#   remove_object    (name)
#   move_object      (name, ...)
#   add_camera       (name, ...)
#   remove_camera    (name)
#   add_robot        (name, ...)
#   remove_robot     (name)
#   list_bodies      (robot_name)        <- asymmetric
#   get_body_state   (body_name)         <- asymmetric; this bug
#   get_robot_state  (robot_name)        <- asymmetric
