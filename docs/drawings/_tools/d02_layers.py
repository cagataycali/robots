"""D2: the seven layers and the import rule. core -> registry -> drivers|mesh -> sim|policies -> app -> tools -> dashboard.

Sources: scripts/check_import_layers.py (LAYERS, KNOWN_DEFERRED_UPWARD_EDGES), docs/concepts/architecture.md
(the layer line the page states), strands_robots/robot.py (the factory in app), strands_robots/tools,
strands_robots/dashboard.
"""

from excal import MUTED, Drawing

d = Drawing(
    "d02_layers",
    "Seven layers read top to bottom: dashboard, tools, app (the Robot factory), sim beside policies, "
    "drivers beside mesh, registry, core. A module imports only from layers below its own; "
    "scripts/check_import_layers.py grades the rule from the source with ast, and every deferred upward "
    "edge is written in its roster, a ratchet that only shrinks.",
)

rows = [
    ("dashboard", "FastAPI + the fleet page, e-stop, consent card", None),
    ("tools", "run_policy, train_policy, pose_tool, use_ros: the agent tool layer", None),
    ("app", 'Robot("so101") the factory; hardware_robot; the operator gate', "accent"),
    (None, None, None),
    ("registry", "robots.json, aliases, drivers, policies.json, assets", None),
    ("core", "envelopes, refusals, audit, units", None),
]
y = 40
boxes = {}
for name, sub, kind in rows:
    if name is None:
        sim = d.box(200, y, 340, 60, "sim", sub="mujoco, newton, isaac: SimEngine", size=16)
        pol = d.box(560, y, 340, 60, "policies", sub="Policy, create_policy, embodiment", size=16)
        boxes["sim"], boxes["policies"] = sim, pol
        y += 90
        drv = d.box(200, y, 340, 60, "drivers", sub="HardwareDriver, feetech, dynamixel, unitree", size=16)
        mesh = d.box(560, y, 340, 60, "mesh", sub="Zenoh peers, security, IoT bridge", size=16)
        boxes["drivers"], boxes["mesh"] = drv, mesh
        y += 90
        continue
    b = d.box(200, y, 700, 60, name, kind=kind or "plain", sub=sub, size=16)
    boxes[name] = b
    y += 90

d.text(40, 60, "imports go", size=13, color=MUTED)
d.text(40, 80, "downward only", size=13, color=MUTED)
d.path([(120, 110), (120, 560)])
d.text(40, 570, "core has no", size=13, color=MUTED)
d.text(40, 590, "strands_robots import", size=13, color=MUTED)

d.text(940, 60, "checked by", size=13, color=MUTED)
d.text(940, 80, "scripts/check_import_layers.py", size=13, color=MUTED)
d.text(940, 100, "with ast, no runtime cycle", size=13, color=MUTED)
d.text(940, 140, "an upward edge must be in", size=13, color=MUTED)
d.text(940, 160, "KNOWN_DEFERRED_UPWARD_EDGES;", size=13, color=MUTED)
d.text(940, 180, "the roster only shrinks", size=13, color=MUTED)

d.caption(200, 690, "sim and policies do not import each other; the app layer joins them in run_policy.")
d.save()
