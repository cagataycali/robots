import ast, pathlib, re, subprocess, sys
ROOT = pathlib.Path(".")
NEW = ["tests/test_inbound_command_costs_one_message.py"]
NEWNAMES = "test_start_poll_is_idempotent"
OLD = ["tests/test_hardware_rtps_bridge.py", "tests/test_hardware_ros_bridge.py",
       "tests/test_ros_telemetry_topic_parity.py",
       "tests/test_ros_telemetry_command_surface_security.py",
       "tests/test_wait_budget_domain.py"]

def fn_range(path, name):
    tree = ast.parse((ROOT/path).read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise SystemExit(f"no {name} in {path}")

def apply(path, fname, old, new):
    """Replace `old` with `new` inside fname only; print in_fn vs in_file."""
    p = ROOT/path; src = p.read_text(); lines = src.splitlines(keepends=True)
    lo, hi = fn_range(path, fname)
    region = "".join(lines[lo-1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{fname}: in_fn={in_fn} in_file={in_file} for {old[:50]!r}"
    print(f"      anchor in_fn={in_fn} in_file={in_file}")
    lines[lo-1:hi] = [region.replace(old, new, 1)]
    out = "".join(lines); ast.parse(out); p.write_text(out)

def run(files, extra=()):
    r = subprocess.run([sys.executable, "-m", "pytest", *files, "-q", "-p", "no:randomly",
                        "--no-cov", *extra], capture_output=True, text=True)
    o = r.stdout
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", o)) else 0
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", o)) else 0
    return f, p

ROS = "strands_robots/ros_telemetry.py"
RTPS = "strands_robots/hardware_rtps_bridge.py"
FIXED = (
    "        action: dict[str, float] = {}\n        for name, pos in zip(names, positions):\n"
    "            try:\n                action[name] = float(pos)\n"
    "            except (TypeError, ValueError):\n"
)
MUTS = [
    ("M1 revert the fix (raise again)", ROS, "_command_action", None),
    ("M2 partially apply instead of rejecting whole", ROS, "_command_action", ("                )\n                return None\n", "                )\n                continue\n")),
    ("M3 refuse but say nothing", ROS, "_command_action", "DROP_LOG"),
    ("M4 narrow the except to ValueError only", ROS, "_command_action", ("            except (TypeError, ValueError):\n", "            except (ValueError,):\n")),
    ("M5 reword the refusal locally", ROS, "_command_action", "REWORD"),
    ("M6 poll loop: no reader tolerance", RTPS, "_poll_loop", ("            except Exception:\n", "            except KeyboardInterrupt:\n")),
    ("M7 _start_poll not idempotent", RTPS, "_start_poll", ("            return\n", "            pass\n")),
]
saves = {p: (ROOT/p).read_text() for p in (ROS, RTPS)}
print(f"{'mutation':46s} {'new':>12s} {'pre-existing':>14s}")
try:
    for label, path, fname, edit in MUTS:
        if edit is None:
            src = (ROOT/path).read_text()
            i = src.index(FIXED); j = src.index("                return None\n", i) + len("                return None\n")
            (ROOT/path).write_text(src[:i] + "        action = {name: float(pos) for name, pos in zip(names, positions)}\n" + src[j:])
            print(f"      anchor in_fn=1 in_file=1 (block)")
        elif edit == "REWORD":
            src = (ROOT/path).read_text()
            i = src.index("                logger.warning(\n")
            j = src.index("                )\n", i) + len("                )\n")
            new_call = (
                "                logger.warning(\n"
                '                    "%s: ignoring a malformed joint_command",\n'
                "                    type(self).__name__,\n"
                "                )\n"
            )
            (ROOT/path).write_text(src[:i] + new_call + src[j:])
            print("      anchor in_fn=1 in_file=1 (reword)")
        elif edit == "DROP_LOG":
            src = (ROOT/path).read_text()
            i = src.index("                logger.warning(\n                    \"%s: ignoring joint_command with a non-numeric")
            j = src.index("                )\n", i) + len("                )\n")
            (ROOT/path).write_text(src[:i] + src[j:])
            print(f"      anchor in_fn=1 in_file=1 (log block)")
        else:
            apply(path, fname, *edit)
        nf, _ = run(NEW + ["tests/test_hardware_rtps_bridge.py"], ("-k", NEWNAMES + " or test_"))
        nf2, _ = run(NEW); nfi, _ = run(["tests/test_hardware_rtps_bridge.py"], ("-k", NEWNAMES))
        of, op = run(OLD, ("-k", f"not {NEWNAMES}"))
        print(f"{label:46s} {nf2+nfi:>7d} failed {of:>7d} failed / {op} passed")
        for p, s in saves.items(): (ROOT/p).write_text(s)
finally:
    for p, s in saves.items(): (ROOT/p).write_text(s)
print("\n=== restored; unmutated control:")
nf2, np2 = run(NEW); nfi, npi = run(["tests/test_hardware_rtps_bridge.py"], ("-k", NEWNAMES)); of, op = run(OLD, ("-k", f"not {NEWNAMES}"))
print(f"  new: {nf2+nfi} failed / {np2+npi} passed    pre-existing: {of} failed / {op} passed")
