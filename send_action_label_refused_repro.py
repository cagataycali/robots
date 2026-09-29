"""Minimal reproducer: send_action refuses joint labels that set_joint_positions accepts.

Registry entry for so101 declares joint_labels {1:shoulder_pan, ..., 6:gripper}. The
docstring on strands_robots.registry.robots.joint_labels says: "so an agent can address
a joint by what it does". get_robot_state() surfaces "1 (shoulder_pan): pos=0.0" as its
text, telling an LLM that either form addresses joint 1.

set_joint_positions honors that contract via _resolve_joint_label
(strands_robots/simulation/mujoco/physics.py:1162). send_action does not: the loop in
_apply_action_by_name (strands_robots/simulation/mujoco/rendering.py:1047) tries the
namespaced actuator, then the namespaced joint, then unresolved -- the label resolver
never runs -- and the refusal advertises only ['1'..'6'] with no mention that labels
exist. Same registry, same robot, two policies on one key.

    $ python send_action_label_refused_repro.py
    set_joint_positions({shoulder_pan: 0.5}) -> success
    send_action({shoulder_pan: 0.5}) -> error: keys ['shoulder_pan'] could not be resolved
    Valid keys hint: ['1', '2', '3', '4', '5', '6']    <- no mention labels exist
"""
from strands_robots import Robot

robot = Robot("so101")

# LABEL WORKS on set_joint_positions
sp = robot(action="set_joint_positions", positions={"shoulder_pan": 0.5}, robot_name="so101")
print("set_joint_positions({shoulder_pan: 0.5}) ->", sp["status"])
assert sp["status"] == "success", sp

# LABEL REFUSED on send_action, on the SAME robot, in the SAME session
sa = robot.send_action({"shoulder_pan": 0.5})
print("send_action({shoulder_pan: 0.5}) ->", sa["status"] + ":", sa["content"][0]["text"][:120])
assert sa["status"] == "error"
assert "shoulder_pan" not in sa["content"][0]["text"] or "Valid keys" not in sa["content"][0]["text"] or "1=" not in sa["content"][0]["text"]

# Verify the labels ARE known to the robot (state text shows them)
state = robot(action="get_robot_state", robot_name="so101")
assert "shoulder_pan" in state["content"][0]["text"], "labels should appear in state text"
print("(state text does mention shoulder_pan, so the LLM will try that key first)")

robot.destroy()
print("REPRO: send_action refuses a label that set_joint_positions accepts, without hinting labels exist.")
