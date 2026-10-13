"""V01: two robots, one world. Today every Robot() is its own engine; in v1 they share Simulation.default()."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v01_one_world",
        "Two robots, one world",
        "A Robot attaches to a Simulation; given none it joins Simulation.default(), so every robot you make shares one world.",
        "Left, today: Robot(\"so101\") and Robot(\"g1\") each return their own MuJoCoSimEngine, two worlds that cannot "
        "see each other, and the dashboard runs a process per card stitched back over the mesh. Right, v1, the one "
        "green element: both robots attach to Simulation.default(), one world holding both bodies, so arm.sim is g1.sim "
        "is True. Under it the backends a world can run on: mujoco, isaac, and physics newton.",
        h=640,
    )
    L, LW = 60, 500
    R, RW = 640, 500

    # ------------------------------------------------------------ today
    s.section(L, 122, "today, v0.5")
    s.box(L, 134, LW, 380, None, None, dashed=True)
    for i, (code, rid) in enumerate([('Robot("so101")', "a"), ('Robot("g1")', "b")]):
        x = L + 20 + i * 240
        s.chip(x, 154, code)
        s.down(x + 60, 178, 214)
        s.box(x, 214, 220, 88, "MuJoCoSimEngine", "a whole world: 8,125 lines, 110 methods, one robot inside",
              size=14, subsize=12)
        s.down(x + 60, 302, 338)
        s.box(x, 338, 220, 64, f"process {i + 1}", "one dashboard card", size=14, subsize=12)
    s.arrow([(L + 140, 426), (L + 380, 426)], head="both", dashed=True, label="stitched over the mesh",
            label_dx=-120, label_dy=22)
    s.para(L + 20, 476, "two worlds that cannot see each other; Isaac allows one SimulationApp per process, "
                        "so there the model cannot even be built", LW - 40, size=12, cls="grot muted")

    # ------------------------------------------------------------ v1
    s.section(R, 122, "v1", accent=True)
    s.chip(R + 20, 154, 'Robot("so101")', id="c_arm")
    s.chip(R + 250, 154, 'Robot("g1", position=[1, 0, 0])', id="c_g1")
    s.arrow([(R + 80, 178), (R + 80, 214)], accent=True, id="w_arm")
    s.arrow([(R + 380, 178), (R + 380, 214)], accent=True, id="w_g1")
    s.box(R, 214, RW, 212, "Simulation.default()",
          "one world: time, scene, physics, cameras, and every body in it", accent=True, size=15, subsize=12.5,
          id="world")
    s.box(R + 20, 286, 220, 64, "so101", "SimBody, fixed base", size=14, subsize=12)
    s.box(R + 260, 286, 220, 64, "unitree_g1", "SimBody, floating base", size=14, subsize=12)
    s.chips(R + 20, 370, ["sim.robots", "sim.time", "sim.render()", "sim.tool"])
    s.text(R + 20, 470, "arm.sim is g1.sim", cls="mono fg", size=15)
    s.text(R + 200, 470, "True", cls="mono accent", size=15)
    s.para(R + 20, 494, "one process, one lock, one scene the dashboard and the agent both read", RW - 40,
           size=12, cls="grot muted")

    s.section(R, 540, "a world runs on")
    s.chips(R, 552, ['"mujoco"', '"isaac"', 'physics="newton"', "num_envs=1024"])
    s.section(L, 540, "what a robot was")
    s.chips(L, 552, ["Simulation | HardwareRobot | HardwareDriver"])

    s.footnote(612, "Simulation.default() is process-global, like asyncio.get_event_loop(); "
                    "Robot(name, sim=Simulation(\"isaac\")) picks another world.")
    s.motion = [("world", "pulse"), ("w_arm", "flow"), ("w_g1", "flow")]
    return s
