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

**Wire-frame contract.** The G1 firmware validates four fields on every
``rt/lowcmd`` frame and silently drops a non-matching one, so the builder
sets them all before publish:

* ``crc`` computed by the SDK's own ``CRC().Crc(cmd)`` after every other
  field is populated - a stale CRC is the single failure mode that looks
  like success from the DDS side and does nothing on the robot.
* ``mode_machine`` echoed from the live ``LowState`` (cached at
  ``G1Driver._mode_machine``, uint8).  A mismatched value is rejected by
  the firmware.  This is **not** the same value as ``G1Driver._fsm_id``,
  which comes from the motion-switcher API and is the high-level state
  the arm-SDK gate tests against - the two fields have disjoint value
  ranges (``[0, 255]`` uint8 vs ``{500, 501, 801}`` SDK constants) and
  conflating them either raises ``struct.error`` on CRC pack or silently
  refuses every real frame.  Until the motion-switcher source is wired
  (harness#361 PR-C, #2765), the gate refuses honestly rather than
  silently rejecting.
* ``mode_pr = 0`` - PR (pitch/roll) mode, which is what the joint-name
  table is calibrated for. AB mode would silently remap four ankle
  indices.
* ``motor_cmd[i].mode = 1`` on every commanded slot - the Enable byte.
  Unset (``0`` = Disable), a frame with a valid CRC still commands
  nothing on that slot. Uncommanded slots stay at 0 (no-op).

Every DDS touch is still lazy: ``unitree_sdk2py`` is imported inside
``send_action`` (the ``LowCmd_`` class) and inside the free
``_build_lowcmd_from_action`` helper (via ``unitree_hg_msg_dds__LowCmd_``
and ``unitree_sdk2py.utils.crc.CRC``), so the driver module still loads
on Thor and on CI.

This is decoupled from issue #358 (the 48 agent-facing @tool verbs); the
driver's own write path only needs one transport primitive, and this PR
supplies it. The 500 Hz control loop lands in the follow-up PR that
closes issue #361 in full.
