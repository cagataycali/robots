---
description: "A UR arm streams joint setpoints over RTDE: the two gates every write passes, how a rollout runs."
---

# Universal Robots e-Series and UR-Series

Here a UR arm streams joint setpoints from `Robot("ur5e", mode="real")` over RTDE; you learn the two gates every write passes and how a policy rollout runs on it.

This needs the arm's controller reachable on the network, Remote Control enabled on the pendant, and the `[ur]` extra:

```bash
pip install 'strands-robots[ur]'          # ur-rtde: rtde_control + rtde_receive
```

```python title="sketch"
from strands_robots import Robot

arm = Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")   # URDriver, RTDE on port 30004; a bare call resolves to lerobot
print(arm.connect_eagerly())
print(arm.state())                                        # q, qd, TCP pose, TCP wrench in one round trip
arm.send_action({"shoulder_pan_joint": 0.0, "wrist_3_joint": 1.57})   # radians, servoJ
```

## Joint names and units

Radians, in the order the wire and the MuJoCo assets share: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`. A recorded sim action indexes onto the wire unchanged. Every joint travels plus or minus 2 pi (`JOINT_LIMIT_RAD`).

## Two gates on every write

A UR controller does not refuse the way a servo bus does: it accepts the register write and does nothing. So the driver checks before it writes:

1. **Controller mode.** `connect_eagerly` opens the receive interface first because it answers whether commanding is possible. A controller in `PROTECTIVE_STOP` accepts an RTDE connection and moves nothing, so the driver refuses there and names the mode.
2. **Step size.** The commanded step is checked against the model's per-joint maximum speed at `control_frequency` (default 125 Hz). A step the arm cannot make in one period is refused, not queued.

`send_action` uses `servoJ` (speed `0.5`, acceleration `0.5`, lookahead `0.1` s, gain `300`) rather than `moveJ`, because a policy streams setpoints and a stream of planned trajectories fights itself.

## Rollouts

`run_policy(policy)` rolls a caller-built policy at `control_frequency`; `start_task(instruction, policy_provider=...)`, which built one in the driver, leaves in 0.8. `get_task_status()` reports the live snapshot, `stop_task()` halts the loop; both share the Feetech driver's rollout class.

## Deliberately absent

| absent | why |
|---|---|
| inverse kinematics, `servoL`, `moveL` | the action space is joint space, the space the policies here emit |
| a gripper | a UR ships without one; `robotiq_2f85` is its own registry entry and driver (Modbus TCP, `Robot("robotiq_2f85", mode="real", port="192.168.1.11")`), paired by composition |
| freedrive, URScript upload | both take the arm out of the RTDE control mode the driver holds |

## Simulation

The driver serves `ur3e`, `ur5e`, `ur7e`, `ur10e`, `ur12e`, `ur16e`, `ur8long`, `ur15`, `ur18`, `ur20`, `ur30` (not CB3), each with a MuJoCo twin. Develop action dicts there; [drivers](drivers.md) explains the shared contract.
