"""V02: the five types. Robot is the centre; Policy beside it; Body under it with three implementations."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v02_five_types",
        "Five types",
        "A Simulation is a world, a Robot is a body in a world or on a bus, a Policy turns an Observation into an Action.",
        "Centre, the one green element: Robot, with observe, act, calibrate, run, start, stop, set_pose and tool. "
        "Left, Policy: act(obs, instruction) returns an Action; implementations are lerobot, onnx, remote, ChunkedPolicy "
        "and any callable. An Action wire runs from Policy to Robot, a dashed Observation wire back. Right, "
        "Description: joints, action_keys, limits, assets and keyframes from the registry, loaded once. Under Robot, "
        "the Body protocol (read, write, pose, images) with three implementations: SimBody into a Simulation, Driver on "
        "a hardware bus, RemoteBody over a Transport. Bottom row: Simulation with its Backend, the feetech and unitree "
        "drivers, and the zenoh and iot transports.",
        h=700,
    )
    # ---------------------------------------------------------------- top row
    s.section(60, 122, "turns observations into actions")
    s.box(60, 134, 290, 170, "Policy", "act(obs, instruction) -> Action; reset(); declares what it needs "
                                       "(cameras, units, gains) before step one", size=15, subsize=12)
    s.chips(74, 236, ["lerobot", "onnx", "remote"])
    s.chips(74, 266, ["ChunkedPolicy", "any callable"])

    s.section(420, 122, "the one type you hold", accent=True)
    s.box(420, 134, 360, 170, "Robot", "the same three verbs on every body; the loop is written once",
          accent=True, size=15, subsize=12, id="robot")
    s.chips(434, 206, ["observe()", "act()", "calibrate()"])
    s.chips(434, 236, ["run()", "start()", "stop()"])
    s.chips(434, 266, ["set_pose()", "tool", "twin"])

    s.section(820, 122, "what the robot is")
    s.box(820, 134, 320, 170, "Description", "from the registry or a path to MJCF or URDF, loaded once; act() "
                                             "validates keys and limits against it", size=15, subsize=12)
    s.chips(834, 236, ["joints", "action_keys", "limits"])
    s.chips(834, 266, ["assets", "keyframes"])

    s.arrow([(350, 196), (420, 196)], label="Action", label_dx=0, label_dy=-8, label_anchor="middle", accent=True, id="act")
    s.arrow([(420, 252), (350, 252)], dashed=True, label="Observation", label_dx=0, label_dy=20, label_anchor="middle", id="obs")
    s.arrow([(820, 219), (780, 219)])

    # ---------------------------------------------------------------- body protocol
    s.down(600, 304, 344, label="robot.body", label_dx=10, label_dy=4, id="to_body")
    s.box(60, 344, 1080, 52, None, None, dashed=True)
    s.text(74, 376, "Body", cls="mono fg", size=15, weight=500)
    s.text(130, 376, "protocol: read() -> JointState, write(targets), pose(), images(), extras(), close()",
           cls="grot text", size=12.5)
    cols = [
        ("SimBody", "a handle into a Simulation: joint and actuator ids by name", 60),
        ("Driver", "a hardware bus; calibrate() reads and writes LeRobot calibration JSON", 434),
        ("RemoteBody", "a robot in another process or on another network, over a Transport", 808),
    ]
    for name, sub, x in cols:
        s.down(x + 166, 396, 420, id=f"to_{name}")
        s.box(x, 420, 332, 84, name, sub, size=14, subsize=12)
        s.motion.append((f"to_{name}", "flow"))
    # ---------------------------------------------------------------- what each body talks to
    s.down(226, 504, 534)
    s.down(600, 504, 534)
    s.down(974, 504, 534)
    s.box(60, 534, 332, 100, "Simulation", "a world; a Backend implements 13 methods", size=14, subsize=12)
    s.chips(74, 596, ["mujoco", "isaac", 'physics="newton"'])
    s.box(434, 534, 332, 100, "hardware", "Robot(\"so101\", port=...) gets a free twin", size=14, subsize=12)
    s.chips(448, 596, ["feetech", "unitree", "microduck"])
    s.box(808, 534, 332, 100, "Transport", "serve(robot, url) and Robot.connect(url)", size=14, subsize=12)
    s.chips(822, 596, ["zenoh://", "iot://"])

    s.footnote(672, "Everything else (recording, agent tools, fleet, dashboard) is written once against these five.")
    s.motion[:0] = [("robot", "pulse"), ("act", "flow"), ("to_body", "flow")]
    return s
