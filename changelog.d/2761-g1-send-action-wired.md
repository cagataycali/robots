### Added: `G1Driver.send_action` writes `LowCmd_` on `rt/lowcmd` (issue #361, PR-B)

The G1 native driver's ``send_action`` verb is wired end-to-end. A caller who
survives the FSM and battery gates now lands one ``LowCmd_`` frame on the
DDS bus instead of a stub "not wired yet" refusal:

```python
robot = Robot("g1", mode="real", driver="strands", port="192.168.1.172",
              network_interface="eth0")
robot.connect_eagerly()

# Scalar targets use the driver's default gains.
robot.send_action({"left_shoulder_pitch": 0.5, "right_elbow": -0.2})

# Per-joint dicts carry gains and feed-forward.
robot.send_action({
    "left_elbow": {"q": 0.3, "kp": 50.0, "kd": 1.5, "dq": 0.1, "tau": 0.05},
})
```

The joint-name -> motor-slot mapping is a module constant
(``_G1_JOINT_INDEX``) so an unknown joint name refuses the whole action -
silent drop is worse than a caller-facing error. Same for an unknown
inner key on a per-joint dict, and for a missing ``q``.

Every DDS touch is still lazy: ``unitree_sdk2py`` is imported inside
``send_action`` (the ``LowCmd_`` class) and inside the free
``_build_lowcmd_from_action`` helper (via ``unitree_hg_msg_dds__LowCmd_``),
so the driver module still loads on Thor and on CI.

This is decoupled from issue #358 (the 48 agent-facing @tool verbs); the
driver's own write path only needs one transport primitive, and this PR
supplies it. The 500 Hz control loop lands in the follow-up PR that
closes issue #361 in full.
