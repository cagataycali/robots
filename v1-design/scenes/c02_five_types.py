"""C02: the five types of v1 and the data that flows between them."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "c02_five_types",
        "Five types",
        "A Description is what a robot is, a Simulation is a world, a Robot is a body, a Policy decides.",
        "Left, Description: joints, action_keys, limits and registry data. In the middle, the one green element, "
        "Robot, with its verbs observe, act, run and its device surface manifest, estop and tool. A Body sits under "
        "the Robot with three implementations: SimBody in a Simulation, a Driver on a bus, and a RemoteBody over the "
        "mesh. Right, Simulation, one world with a Backend: MuJoCo, Isaac or Newton. Above the Robot, Policy: a "
        "callable, an ONNX skill, a LeRobot checkpoint or a LeRobot server over Zenoh. Observation flows up from the "
        "Robot to the Policy and Action flows back down. A row of chips names the four data types at the bottom.",
        h=660,
    )
    s.section(60, 120, "what a robot is")
    s.box(60, 132, 250, 190, "Description", "read from the model file and the registry", size=14, subsize=12, id="desc")
    s.chips(74, 206, ["joints", "action_keys"])
    s.chips(74, 242, ["limits", "keyframes"])
    s.chips(74, 278, ["hardware", "device.type"])

    s.section(410, 120, "the brain")
    s.box(410, 132, 380, 120, 'Policy(spec)', "picked by the form of spec, never by a model family", size=14,
          subsize=12, id="policy")
    s.chips(424, 206, ["f(obs)", '"microduck?skill=…"', '"lerobot/…"', '"zenoh://…"'])

    s.section(410, 296, "the one type you hold", accent=True)
    s.box(410, 308, 380, 150, "Robot", "the same verbs in a world and on a bus", accent=True, size=15, subsize=12,
          id="robot")
    s.chips(424, 382, ["observe()", "act()", "run()", "start()"])
    s.chips(424, 418, ["manifest", "estop()", "tool"])
    s.arrow([(530, 308), (530, 252)], label="Observation", label_dx=8, label_dy=4, id="obs")
    s.arrow([(670, 252), (670, 308)], label="Action", label_dx=8, label_dy=4, id="act")
    s.arrow([(310, 383), (410, 383)], label="describes", label_dx=-40, label_dy=-8, id="d2r")

    s.section(410, 482, "the body: the only thing that differs")
    s.box(410, 494, 116, 64, "SimBody", "in a world", size=12, subsize=11, id="b1")
    s.box(542, 494, 116, 64, "Driver", "on a bus", size=12, subsize=11, id="b2")
    s.box(674, 494, 116, 64, "RemoteBody", "over the mesh", size=12, subsize=11, id="b3")
    s.arrow([(600, 458), (600, 494)], id="r2b")

    s.section(890, 120, "the world")
    s.box(890, 132, 250, 326, "Simulation", "owns time and the scene; every Robot without sim= joins the default one",
          size=14, subsize=12, id="sim")
    s.chips(904, 222, ["Simulation.default()"])
    s.chips(904, 258, ["Backend: 13 methods"])
    s.chips(904, 294, ["mujoco", "isaac"])
    s.chips(904, 330, ['physics="newton"'])
    s.chips(904, 366, ["sim.run({...})"])
    s.chips(904, 402, ["sim.tool"])
    s.arrow([(526, 558), (526, 590), (1015, 590), (1015, 458)], dashed=True, label="steps", label_dx=10,
            label_dy=-6, id="b2s")
    s.footnote(626, "strands_robots.__all__ is exactly Robot, Simulation, Policy, Body, Description, Observation, "
                    "Action, JointState, Pose; everything else is imported from its package.")
    s.motion = [("robot", "pulse"), ("obs", "flow"), ("act", "flow"), ("d2r", "flow"), ("r2b", "flow"),
                ("b2s", "flow")]
    return s
