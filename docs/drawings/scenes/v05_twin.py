"""V05: a real robot brings its twin; the operator gate sits in front of the bus."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v05_twin",
        "A real robot brings its twin",
        "Robot(port=) drives the bus and mirrors every command into a twin in Simulation.default(); the dashboard draws the twin.",
        "Left, the one green element: Robot(\"so101\", port=...), a hardware robot. A wire runs right through the "
        "operator gate, confirm= a callable the operator owns (the model's confirm=true only unlocks it, never "
        "replaces it), to the feetech Driver and the STS3215 bus. A second wire runs down to so101_twin, a SimBody in "
        "Simulation.default(): act(a) mirrors a into the twin, observe() pushes the measured joints into it, so "
        "real.twin.sim.render() shows what the arm does. calibrate() reads and writes LeRobot calibration JSON.",
        h=640,
    )
    s.section(60, 122, "the robot you hold", accent=True)
    s.box(60, 134, 340, 132, 'Robot("so101", port=...)', "a hardware robot; the same observe, act, calibrate and "
                                                        "run as in a world", accent=True, size=14, subsize=12,
          id="real")
    s.chips(74, 226, ["is_real", "twin", "calibrate()"])

    s.section(460, 122, "the operator gate")
    s.box(460, 134, 300, 132, "confirm=", "a callable the operator owns; tools and the dashboard reach the bus "
                                         "only through it",
          size=14, subsize=12, id="gate")
    s.section(820, 122, "the bus")
    s.box(820, 134, 320, 132, "Driver: feetech", "read() and write() on the STS3215 servos; calibration is "
                                                 "LeRobot's file format, unchanged", size=14, subsize=12, id="bus")
    s.chips(834, 226, ["/dev/tty.usbmodem...", "unitree"])
    s.arrow([(400, 200), (460, 200)], label="act(a)", label_dx=-22, label_dy=-8, id="w1")
    s.arrow([(760, 200), (820, 200)], label="targets", label_dx=-24, label_dy=-8, id="w2")

    # twin
    s.arrow([(160, 266), (160, 380)], label="mirrors act(a)", label_dx=10, label_dy=4, id="mirror")
    s.arrow([(300, 380), (300, 266)], dashed=True, label="observe(): measured joints", label_dx=10, label_dy=4,
            id="measured")
    s.section(440, 368, "the twin, in the default world")
    s.box(60, 380, 640, 156, None, None, dashed=True)
    s.box(80, 400, 280, 84, "so101_twin", "a SimBody of the same description; twin.twin_of is the real robot",
          size=14, subsize=12, id="twin")
    s.box(380, 400, 300, 84, "real.twin.sim.render()", "what the dashboard and the agent see", size=14,
          subsize=12)
    s.para(80, 508, "off with twin=False; a Robot in an explicit Simulation gets no twin, it is already in a world",
           600, size=12, cls="grot muted")

    s.section(760, 368, "refused")
    s.box(760, 380, 380, 156, "operator says no", "nothing moves: the tool returns the refusal, the bus is never "
                                                  "written, the twin is not mirrored", size=14, subsize=12)
    s.chips(774, 496, ["set_pose() raises", "sim=  raises"])

    s.footnote(590, "A hardware act() costs wall time; a world act() costs one control step. The loop is the same "
                    "line of code on both.")
    s.motion = [("real", "pulse"), ("w1", "flow"), ("w2", "flow"), ("mirror", "flow"), ("measured", "flow")]
    return s
