"""D2: the seven layers and the import rule.

core -> registry -> drivers | mesh -> sim | policies -> app -> tools -> dashboard. A module imports only
from layers below its own; scripts/check_import_layers.py grades the rule with ast and every deferred
upward edge sits in KNOWN_DEFERRED_UPWARD_EDGES, a roster that only shrinks. The Robot factory lives in
app, the one layer that sees both sim and policies.
"""
from scene import Scene

L, LW = 60, 620          # the stack
R, RW = 780, 360         # the rule
BUS = 720                # the downward wire beside the stack


def scene() -> Scene:
    s = Scene(
        "d02_layers",
        "Seven layers, imports point down",
        "Read the package top to bottom; a module imports only from the layers under it, and one script grades that from the source.",
        "Left, a stack of seven layers read top to bottom: dashboard, tools, app (the Robot factory, the one "
        "green element), sim beside policies, drivers beside mesh, registry, core. A wire beside the stack "
        "points downward, labelled imports. Right, three cards: check_import_layers reads every import with "
        "ast; deferred upward edges must be in KNOWN_DEFERRED_UPWARD_EDGES, a roster that only shrinks; the "
        "factory in app joins sim and policies in run_policy, and the two never import each other. Footnote: "
        "core has no strands_robots import at all.",
        h=700,
    )
    s.section(L, 122, "the stack, top to bottom")
    rows = [
        ("dashboard", "FastAPI and the fleet page, the consent card, e-stop"),
        ("tools", "run_policy, train_policy, pose_tool, use_ros: what an agent calls"),
        ("app", 'Robot("so101"), the factory; hardware_robot; the operator gate', True),
        None,
        None,
        ("registry", "robots.json, aliases, drivers, policies.json, assets"),
        ("core", "envelopes, refusals, audit, units"),
    ]
    y, rh, gap = 134, 60, 12
    half = (LW - 12) / 2
    pairs = iter([
        (("sim", "mujoco, newton, isaac: SimEngine"), ("policies", "Policy, create_policy, EmbodimentMap")),
        (("drivers", "HardwareDriver: feetech, dynamixel, unitree"), ("mesh", "Zenoh peers, security, the IoT bridge")),
    ])
    for row in rows:
        if row is None:
            (t1, s1), (t2, s2) = next(pairs)
            s.box(L, y, half, rh, t1, s1, size=14, subsize=12)
            s.box(L + half + 12, y, half, rh, t2, s2, size=14, subsize=12)
        else:
            title, sub, *acc = row
            s.box(L, y, LW, rh, title, sub, accent=bool(acc), size=14, subsize=12)
        y += rh + gap
    bottom = y - gap
    s.arrow([(BUS, 134), (BUS, bottom)], label="imports", label_dx=10, label_dy=4)

    s.section(R, 122, "the rule")
    s.box(R, 134, RW, 118, "check_import_layers.py",
          "reads every import in strands_robots with ast, no runtime cycle involved; an import that points "
          "at a higher layer is an inversion and fails the run", size=14, subsize=12)
    s.chips(R + 14, 218, ["scripts/check_import_layers.py"])
    s.box(R, 268, RW, 118, "deferred upward edges",
          "an import inside a function still points up; it is allowed only when the module pair is in "
          "the roster, and the roster only shrinks", size=14, subsize=12)
    s.chips(R + 14, 352, ["KNOWN_DEFERRED_UPWARD_EDGES"])
    s.box(R, 402, RW, 118, "where the two halves meet",
          'sim and policies never import each other; Robot("so101") in app joins them inside run_policy, '
          "and the tools above it never see either", size=14, subsize=12)
    s.chips(R + 14, 486, ["run_policy", "create_policy"])
    s.box(R, 536, RW, 102, "why it holds",
          "a driver can be tested without a simulator, a policy without a robot, and core has nothing "
          "of strands_robots above it to import", size=14, subsize=12)

    s.footnote(672, "core has no strands_robots import at all; every layer above it can be replaced without touching it.")
    return s
