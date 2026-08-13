"""Mutation table: 2 arms x 5 plausible regressions."""
import ast, pathlib, re, subprocess, sys
SRC = pathlib.Path("strands_robots/simulation/isaac/motion_primitives.py")
ORIG = SRC.read_text()
NEW_NAMES = [
    "test_refused_while_a_policy_runs_on_the_robot",
    "test_the_policy_refusal_is_one_rule_across_the_primitives",
    "test_a_policy_on_another_robot_does_not_refuse",
    "test_an_uninitialized_articulation_reports_its_own_reason",
    "test_a_policy_started_mid_run_aborts",
    "test_the_mid_run_policy_abort_is_one_rule_across_the_primitives",
    "test_a_removed_robot_reports_removal_not_a_policy",
]
K_NEW = " or ".join(NEW_NAMES)
K_OLD = f"not ({K_NEW})"

UPFRONT = '''        if robot.policy_running:
            return (
                None,
                None,
                _err(
                    f"Cannot '{action}' on '{robot_name}' while its policy is running - a primitive "
                    "and the policy loop would race on the articulation's PD targets. Wait "
                    "for the rollout to finish (Isaac policy loops clear the flag on exit)."
                ),
            )
'''
ART = '''        if robot.articulation is None:
            return None, None, _err(f"Robot {robot_name!r} not initialized.")
'''
MIDRUN = '''        if robot.policy_running:
            return _err(f"{action}: a policy started on '{robot_name}' mid-run; aborting.")
'''
REMOVED = '''        if robot is None or robot.articulation is None:
            return _err(f"{action}: robot '{robot_name}' was removed mid-run; aborting.")
'''

def fn_range(text, name):
    t = ast.parse(text)
    for cls in [n for n in t.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name == name:
                return fn.lineno, fn.end_lineno
    raise AssertionError(name)

def scoped(text, fn, old, new):
    """Replace `old` with `new` inside function `fn` only, asserting uniqueness."""
    lo, hi = fn_range(text, fn)
    lines = text.splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    in_fn, in_file = region.count(old), text.count(old)
    assert in_fn == 1, f"{fn}: in_fn={in_fn} in_file={in_file}"
    print(f"    anchor in_fn={in_fn} in_file={in_file} ({fn})")
    lines[lo - 1 : hi] = [region.replace(old, new, 1)]
    out = "".join(lines)
    assert out != text
    return out

MUTATIONS = {
    "M1 drop the up-front policy guard": lambda s: scoped(s, "_primitive_resolve_robot", UPFRONT, ""),
    "M2 drop the mid-run policy abort": lambda s: scoped(s, "_primitive_abort_reason", MIDRUN, ""),
    "M3 policy guard before the articulation check": lambda s: scoped(
        s, "_primitive_resolve_robot", ART + UPFRONT, UPFRONT + ART
    ),
    "M4 policy abort before the removed-robot check": lambda s: scoped(
        s, "_primitive_abort_reason", REMOVED + MIDRUN, MIDRUN + REMOVED
    ),
    "M5 global scope: any robot's policy refuses": lambda s: scoped(
        s,
        "_primitive_resolve_robot",
        "        if robot.policy_running:\n",
        "        if any(r.policy_running for r in self._robots.values()):\n",
    ),
    "M6 reword the refusal locally in the preamble": lambda s: scoped(
        s,
        "_primitive_resolve_robot",
        '''                _err(
                    f"Cannot '{action}' on '{robot_name}' while its policy is running - a primitive "
                    "and the policy loop would race on the articulation's PD targets. Wait "
                    "for the rollout to finish (Isaac policy loops clear the flag on exit)."
                ),
''',
        '''                _err(f"{action}: busy."),
''',
    ),
}

def run(k):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/simulation/isaac/test_motion_primitives.py", "-q", "--no-cov",
         "-p", "no:randomly", "-k", k],
        capture_output=True, text=True,
    )
    m = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return int(m.group(1)) if m else 0, int(p.group(1)) if p else 0

print("control (unmutated):")
for label, k in (("new", K_NEW), ("pre-existing", K_OLD)):
    f, p = run(k)
    print(f"  {label:14s} {f} failed / {p} passed")

rows = []
for label, fn in MUTATIONS.items():
    print(f"\n{label}")
    try:
        SRC.write_text(fn(ORIG))
        fn_new, pn = run(K_NEW)
        fo, po = run(K_OLD)
        rows.append((label, fn_new, pn, fo, po))
        print(f"    new: {fn_new} failed / {pn} passed   |   pre-existing: {fo} failed / {po} passed")
    finally:
        SRC.write_text(ORIG)
assert SRC.read_text() == ORIG
print("\n=== TABLE ===")
for label, fn_new, pn, fo, po in rows:
    print(f"{label:48s} new={fn_new:>2} failed  pre-existing={fo:>2} failed")
print(f"\ncaught by the new cases: {sum(1 for r in rows if r[1] > 0)}/{len(rows)}")
print(f"invisible to pre-existing: {sum(1 for r in rows if r[3] == 0)}/{len(rows)}")
