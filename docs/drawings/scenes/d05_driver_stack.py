"""D5: one interface, the backends under it. Robot -> engine or driver -> transport -> motors.

Sources: strands_robots/robot.py (factory, mode, driver=, transport="twin"), strands_robots/simulation
(mujoco, newton, isaac backends), hardware_robot.py (the lerobot driver), strands_robots/drivers (native
drivers and their transports).
"""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d05_driver_stack",
        "One interface, several backends under it",
        "get_observation, send_action, run_policy and the agent tool sit on a simulator or a driver; the driver reaches the motors over a transport.",
        "Top, the one green element: one interface, get_observation, send_action, run_policy and the agent "
        "tool. Two dashed layers under it. mode=\"sim\": MuJoCo on the CPU, the default; Newton and Isaac Sim "
        "on a GPU; one registry entry and the same MJCF assets, nothing gated in simulation; a twin card, "
        "transport=\"twin\", the native driver steps the MuJoCo model instead of a bus. mode=\"real\": the "
        "lerobot driver, the default, or a native driver; under them the transports serial, TCP, DDS and ROS "
        "2, and under those the motors, the only thing the gate protects. Footnote: the call does not "
        "change; what refuses it does.",
        h=672,
    )
    s.box(300, 122, 600, 78, "one interface",
          "get_observation, send_action, run_policy and the agent tool: the same verbs on every backend",
          accent=True, size=14, subsize=12)
    s.arrow([(600, 200), (600, 226), (310, 226), (310, 250)])
    s.arrow([(600, 226), (890, 226), (890, 250)])
    s.text(455, 220, 'mode="sim"', cls="mono muted", size=10.5, anchor="middle")
    s.text(745, 220, 'mode="real"', cls="mono muted", size=10.5, anchor="middle")

    # ---------------------------------------------------------------- simulation
    s.box(60, 250, 500, 350, None, None, dashed=True)
    s.text(74, 272, 'MODE="SIM": AN ENGINE, NEVER GATED', cls="mono muted", size=10.5, spacing="0.05em")
    s.box(80, 290, 145, 88, "MuJoCo", "CPU, the default", size=14, subsize=12)
    s.box(240, 290, 145, 88, "Newton", 'GPU, backend="newton"', size=14, subsize=12)
    s.box(400, 290, 140, 88, "Isaac Sim", "GPU, a plugin", size=14, subsize=12)
    s.para(80, 404, "one registry entry and the same MJCF assets on every engine; add_robot, add_camera and "
           "add_object grow the scene", 460, size=12)
    s.chips(80, 444, ["add_robot", "add_camera", "add_object", "render"])
    s.box(80, 494, 460, 92, "twin",
          'transport="twin": the native driver steps the MuJoCo model instead of a bus, a rehearsal of the real call',
          size=14, subsize=12)

    # ---------------------------------------------------------------- hardware
    s.box(620, 250, 520, 350, None, None, dashed=True)
    s.text(634, 272, 'MODE="REAL": A DRIVER, A TRANSPORT, THE MOTORS', cls="mono muted", size=10.5, spacing="0.05em")
    s.box(640, 290, 230, 88, "lerobot driver", 'driver="lerobot", the default: lerobot robot classes on a USB port',
          size=14, subsize=12)
    s.box(890, 290, 230, 88, "native driver", 'driver="strands": Feetech, Dynamixel, Unitree, Franka',
          size=14, subsize=12)
    s.text(716, 412, "TRANSPORT", cls="mono muted", size=10.5, spacing="0.05em")
    tx = [(640, 112, "serial", "Feetech, Dynamixel"), (760, 112, "TCP", "Franka, Robotiq"),
          (880, 112, "DDS", "Unitree"), (1000, 120, "ROS 2", "any ROS stack")]
    for x, w, t, sub in tx:
        s.box(x, 420, w, 74, t, sub, size=13, subsize=10.5)
        s.down(x + w / 2, 494, 518)
    s.down(696, 378, 420)
    s.arrow([(1005, 378), (1005, 400), (816, 400), (816, 420)])
    s.arrow([(1005, 400), (936, 400), (936, 420)])
    s.arrow([(1005, 400), (1060, 400), (1060, 420)])
    s.box(640, 518, 480, 68, "motors",
          "the only thing the gate protects: a real send_action ends here, a twin's ends in the model",
          size=14, subsize=12)

    s.footnote(644, "the call does not change; what refuses it does: a missing extra, a busy bus, an operator who said no.")
    return s
