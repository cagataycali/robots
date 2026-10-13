"""V06: transports and the fleet. Any robot anywhere is a Robot; Fleet finds them; use_fleet hands them to an agent."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v06_fleet",
        "Any robot, anywhere, the same three verbs",
        "Transport.serve exposes a robot; Robot.connect returns a Robot whose Body is a RemoteBody; Fleet finds them all.",
        "Left column, three robots: one in this process, one served over zenoh, one served as an AWS IoT thing. "
        "Middle, two transports: zenoh (strands scouting group 224.0.0.224:7447, act round trip 0.45 ms) and AWS IoT "
        "Core Direct Messaging (commands are confirmed direct messages, act 189 to 500 ms, 404 means offline and "
        "nothing is queued; state is MQTT). Right, the one green element: Fleet.discover(), one wildcard describe "
        "query plus the local robots plus retained IoT describes, returning Devices; under it use_fleet with list, "
        "describe, observe, act, run and stop, handed to a strands Agent.",
        h=640,
    )
    s.section(60, 122, "robots")
    s.box(60, 134, 300, 96, "this process", "Simulation.default().robots; no transport at all", size=14, subsize=12)
    s.box(60, 250, 300, 96, 'serve(arm, "zenoh://strands/so101")', "another process, same LAN", size=13,
          subsize=12)
    s.box(60, 366, 300, 96, 'serve(g1, "iot://g1-lab")', "anywhere with internet; the key never leaves the robot",
          size=13, subsize=12)

    s.section(420, 122, "transports")
    s.box(420, 250, 340, 96, "zenoh", "peers scout on 224.0.0.224:7447, a strands group, not the default",
          size=14, subsize=12)
    s.box(420, 366, 340, 96, "IoT Core Direct Messaging", "a confirmed message per command; 404 means offline, "
                                                          "nothing is queued", size=14, subsize=12)
    s.arrow([(360, 298), (420, 298)], id="z1")
    s.arrow([(360, 414), (420, 414)], id="i1")
    s.chips(420, 474, ["act 0.45 ms", "act 189 to 500 ms", "connect 859 ms"])

    s.section(820, 122, "one view of all of them", accent=True)
    s.box(820, 134, 320, 212, "Fleet.discover()", "one wildcard query on <ns>/*/describe, plus this process, plus "
                                                  "retained IoT describes; every Device connects to a Robot",
          accent=True, size=14, subsize=12, id="fleet")
    s.chips(834, 250, ["Device", "fleet.connect(id)"])
    s.chips(834, 280, ["Fleet(confirm=)"])
    s.arrow([(360, 182), (820, 182)], id="l1")
    s.arrow([(760, 298), (820, 298)], id="z2")
    s.arrow([(760, 414), (790, 414), (790, 320), (820, 320)], id="i2")

    s.down(980, 346, 380, id="tool")
    s.box(820, 380, 320, 140, "use_fleet", "the agent tool: an act on a real robot passes that robot's own "
                                           "operator gate", size=14, subsize=12)
    s.chips(834, 454, ["list", "describe", "observe"])
    s.chips(834, 484, ["act", "run", "stop"])

    s.section(60, 498, "the same robot, remote")
    s.box(60, 510, 700, 56, None, None, dashed=True)
    s.text(74, 543, 'remote = Robot.connect("zenoh://strands/so101");  remote.act(remote.observe().joints)',
           cls="mono fg", size=12.5)

    s.footnote(604, "The e-stop is a direct message too: it reaches a connected robot or fails at once, never a "
                    "retained command that fires on a robot booting later.")
    s.motion = [("fleet", "pulse"), ("l1", "flow"), ("z1", "flow"), ("z2", "flow"), ("i1", "flow"), ("i2", "flow"),
                ("tool", "flow")]
    return s
