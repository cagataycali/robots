"""C08: the parts of v1 and where each one is: merged on dev, open as a PR, or waiting."""
from scene import Scene

PARTS = [
    ("0", "empty dev", "merged #4754"), ("1", "types + Description", "merged #4755"),
    ("2", "Simulation + MuJoCo", "merged #4756"), ("3", "Robot is a class", "merged #4757"),
    ("3.5", "one concept, one package", "merged #4772"), ("4", "Policy", "PR #4773"),
    ("5", "drivers + twin + gate", "merged #4758"), ("5.5", "a Robot is a device", "PR #4774"),
    ("6", "Isaac + Newton", "PR #4775"), ("7", "the mesh", "PR #4776"),
    ("8", "agent tools", "PR #4778"), ("9", "recording", "PR #4777"),
    ("10", "dashboard", "PR #4779"), ("acc", "acceptance suite", "PR #4780 (+ #4771)"),
    ("12", "docs + notebooks", "PR #4781"), ("11", "dev becomes main", "after the stack merges"),
]


def scene() -> Scene:
    s = Scene(
        "c08_parts",
        "v1, part by part",
        "Each part is one pull request with a gate. Merged parts are on dev; the open ones are a stack, merged "
        "lowest first; dev becomes main last.",
        "Sixteen cards in four rows, lit one after another. Merged on dev: 0 empty dev, 1 types and Description, "
        "2 Simulation and MuJoCo, 3 Robot is a class, 3.5 one concept one package, 5 drivers twin and gate. Open "
        "as stacked pull requests: 4 Policy, 5.5 a Robot is a device, 6 Isaac and Newton, 7 the mesh, 8 agent "
        "tools, 9 recording, 10 the dashboard, the acceptance suite and 12 docs and notebooks. Part 11, dev "
        "becoming main, waits for the stack. Part 4, the one green element, is the next to merge.",
        h=700,
    )
    X, W, GAP = 60, 1080, 20
    bw = (W - GAP * 3) / 4
    bh = 92
    for i, (n, t, st) in enumerate(PARTS):
        r, c = divmod(i, 4)
        x, y = X + c * (bw + GAP), 132 + r * (bh + 34)
        s.box(x, y, bw, bh, f"{n}. {t}", st, accent=(n == "4"), dashed=st.startswith("after"), size=13,
              subsize=12, id=f"p{i}")
        s.motion.append((f"p{i}", "visit"))
    s.section(X, 640, "merge rule")
    s.chips(X + 120, 628, ["lowest open PR first", "reviewer approves after the last push", "required check green",
                           "squash"])
    s.footnote(676, "Every other driver and policy provider comes after 1.0, on these interfaces, inside "
                    "strands_robots.")
    return s
