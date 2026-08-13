"""Mutation table: do the new cases catch a regression the pre-existing ones miss?

Each anchor is scoped to run_multi_policy's own AST line range; in_fn vs
in_file is printed as the justification for that scoping.
"""
import ast, json, pathlib, re, shutil, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/isaac/simulation.py")
TEST = pathlib.Path("tests/simulation/isaac/test_run_multi_policy_no_recording.py")
NEW = ["test_run_multi_policy_rejects_a_frequency_it_cannot_divide_by",
       "test_run_multi_policy_rejects_a_duration_it_cannot_honor",
       "test_run_multi_policy_checks_duration_only_when_it_is_the_effective_horizon",
       "test_run_multi_policy_rejects_instructions_the_shared_helper_refuses",
       "test_run_multi_policy_rejects_an_action_horizon_the_shared_helper_refuses"]

# derive the two -k expressions from the file itself
tree = ast.parse(TEST.read_text())
all_tests = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
assert set(NEW) <= set(all_tests), set(NEW) - set(all_tests)
K_NEW = " or ".join(NEW)
K_OLD = f"not ({K_NEW})"

src = SRC.read_text()
fn = next(n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "run_multi_policy")
lines = src.splitlines(keepends=True)
region = "".join(lines[fn.lineno - 1: fn.end_lineno])

FREQ = "        if err := self._validate_positive_frequency(control_frequency, \"run_multi_policy\"):\n            return err\n"
DUR = "        if n_steps is None:\n            if err := self._validate_duration(duration, \"run_multi_policy\"):\n                return err\n"
INSTR = ("        instr_map, err = self._normalize_multi_policy_instructions(\n"
         "            policies, instructions, \"run_multi_policy\", warn_logger=logger\n"
         "        )\n        if err is not None:\n            return err\n")
HORIZ = ("        horizon_map, err = self._normalize_multi_policy_horizons(\n"
         "            policies, action_horizon, \"run_multi_policy\", default_horizon=8\n"
         "        )\n        if err is not None:\n            return err\n")

RESOLVE = '        duration, n_steps, horizon_error = self._resolve_horizon(\n            n_steps, max_steps, control_frequency, duration, "run_multi_policy"\n        )\n        if horizon_error is not None:\n            return horizon_error\n'

MUTS = [
    ("M1 frequency guard deleted", FREQ, ""),
    ("M2 frequency verdict discarded", FREQ,
     "        self._validate_positive_frequency(control_frequency, \"run_multi_policy\")\n"),
    ("M3 duration guard deleted", DUR, ""),
    ("M4 duration gate removed (behaviour-preserving)", DUR,
     "        if err := self._validate_duration(duration, \"run_multi_policy\"):\n            return err\n"),
    ("M5 instructions verdict discarded", INSTR,
     "        instr_map, err = self._normalize_multi_policy_instructions(\n"
     "            policies, instructions, \"run_multi_policy\", warn_logger=logger\n"
     "        )\n        instr_map = instr_map or {}\n"),
    ("M6 action_horizon verdict discarded", HORIZ,
     "        horizon_map, err = self._normalize_multi_policy_horizons(\n"
     "            policies, action_horizon, \"run_multi_policy\", default_horizon=8\n"
     "        )\n        horizon_map = horizon_map or dict.fromkeys(policies, 8)\n"),
    ("M8 caller argument validated instead of resolved horizon",
     RESOLVE + DUR,
     "        if err := self._validate_duration(duration, \"run_multi_policy\"):\n            return err\n" + RESOLVE),
    ("M7 message re-worded locally (wrong method name)", FREQ,
     "        if err := self._validate_positive_frequency(control_frequency, \"run_policy\"):\n            return err\n"),
]

def run(kexpr):
    p = subprocess.run([sys.executable, "-m", "pytest", str(TEST), "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no", "-k", kexpr],
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in p.stdout.splitlines() if re.match(r"^={5,}.*(passed|failed|error)", l)]
    txt = tail[-1] if tail else p.stdout.strip().splitlines()[-1:]
    def g(pat):
        m = re.search(pat, str(txt)); return int(m.group(1)) if m else 0
    return g(r"(\d+) failed"), g(r"(\d+) passed"), g(r"(\d+) error")

backup = SRC.with_suffix(".bak"); shutil.copy2(SRC, backup)
rows = []
try:
    print("=== unmutated control ===")
    for label, k in (("new", K_NEW), ("pre-existing", K_OLD)):
        f, p, e = run(k); print(f"  {label:12s} failed={f} passed={p} errors={e}")
        assert f == 0 and e == 0, (label, f, e)

    for label, old, new in MUTS:
        in_fn, in_file = region.count(old), src.count(old)
        assert in_fn == 1, (label, "in_fn", in_fn)
        mutated = src.replace(old, new, 1)
        assert mutated != src, label
        SRC.write_text(mutated)
        try:
            fn_new, pn_new, en_new = run(K_NEW)
            fn_old, pn_old, en_old = run(K_OLD)
        finally:
            shutil.copy2(backup, SRC)
        rows.append({"label": label, "in_fn": in_fn, "in_file": in_file,
                     "new_failed": fn_new + en_new, "old_failed": fn_old + en_old})
        print(f"{label:48s} in_fn={in_fn} in_file={in_file}  new={fn_new + en_new:2d} failed   "
              f"pre-existing={fn_old + en_old:2d} failed")
finally:
    shutil.copy2(backup, SRC); backup.unlink()
    assert SRC.read_text() == src, "restore is not byte-identical"

caught = sum(r["new_failed"] > 0 for r in rows)
blind = sum(r["new_failed"] > 0 and r["old_failed"] == 0 for r in rows)
print(f"\ncaught by the new cases: {caught}/{len(rows)}   invisible to the pre-existing: {blind}/{len(rows)}")
pathlib.Path(f"/tmp/mut-{__import__('os').environ['GITHUB_RUN_ID']}.json").write_text(json.dumps(rows, indent=1))
