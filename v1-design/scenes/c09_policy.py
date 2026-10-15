"""C09: Policy(spec) picks a provider from the form of spec; LeRobot runs natively, in-process or over Zenoh."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "c09_policy",
        "Any policy, any robot",
        "Policy(spec) picks the provider from the form of spec. LeRobot is spoken natively, here or over Zenoh.",
        "Left, the one green element: Policy(spec). Four wires go right to four providers: a callable f(obs), an "
        "ONNX skill such as holosoma or microduck with skill velstand, roulade or alpha_ground_pick, a LeRobot "
        "checkpoint loaded in this process by LeRobot's load_deployment, and a LeRobot policy server on a GPU "
        "reached over Zenoh by LeRobot's RemoteClient. The two LeRobot providers pass through a dashed box, the "
        "contract: joint names, cameras, units from semantics.json, refused by name before a session exists. All "
        "four produce an Action that the one run loop sends to the robot.",
        h=660,
    )
    s.section(60, 120, "you write", accent=True)
    s.box(60, 132, 260, 400, "Policy(spec)", "an unknown form is refused, listing the forms", accent=True,
          size=14, subsize=12, id="spec")
    s.chips(74, 214, ["lambda obs: {...}"])
    s.chips(74, 300, ['"holosoma"'])
    s.chips(74, 336, ['"microduck?skill=roulade"'])
    s.chips(74, 410, ['"lerobot/smolvla_base"'])
    s.chips(74, 480, ['"zenoh://gpu:7447/smolvla"'])

    s.section(400, 120, "providers")
    s.box(400, 132, 330, 70, "CallablePolicy", "called every step", size=13, subsize=11, id="p1")
    s.box(400, 222, 330, 120, "ONNX skills", "presets.json: holosoma (G1), ten microduck skills; sha256-checked",
          size=13, subsize=11, id="p2")
    s.box(400, 362, 330, 70, "LeRobotPolicy + ChunkedPolicy", "LeRobot's own load_deployment", size=13, subsize=11,
          id="p3")
    s.box(400, 452, 330, 80, "ZenohPolicy", "LeRobot's RemoteClient to any lerobot-policy-server", size=13,
          subsize=11, id="p4")
    for y, i in ((167, 1), (282, 2), (397, 3), (492, 4)):
        s.arrow([(320, y), (400, y)], id=f"w{i}")

    s.section(800, 352, "the edge (LeRobot only)")
    s.box(800, 362, 150, 170, "contract", "names, cameras, units; refused by name", dashed=True, size=13, subsize=11,
          id="edge")
    s.arrow([(730, 397), (800, 397)], id="e1")
    s.arrow([(730, 492), (800, 492)], id="e2")

    s.section(1000, 120, "out")
    s.box(1000, 132, 140, 400, "Action", "the one run loop sends it to the robot", size=14, subsize=12, id="action")
    s.arrow([(730, 167), (1000, 167)], id="o1")
    s.arrow([(730, 282), (1000, 282)], id="o2")
    s.arrow([(950, 447), (1000, 447)], id="o3")
    s.footnote(580, "Measured on an L40S: 13 LeRobot policy families admitted and driving a sim robot over Zenoh, "
                    "the same ACT checkpoint in-process and served gives the same actions (max difference 0.0).")
    s.motion = [("spec", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow"), ("w4", "flow"), ("e1", "flow"),
                ("e2", "flow"), ("o1", "flow"), ("o2", "flow"), ("o3", "flow")]
    return s
