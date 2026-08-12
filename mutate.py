"""Mutation table: which regressions do the new tests catch, and which does main's suite?"""
import ast, json, os, pathlib, re, subprocess, sys

RID = os.environ["GITHUB_RUN_ID"]
NEW = "tests/policies/wbc/test_wbc_torque_control_lifecycle.py"
SIM = pathlib.Path("strands_robots/policies/wbc/sim_control.py")
HOOK = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")

def fn_range(path, name):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise SystemExit(f"no {name} in {path}")

MUTATIONS = [
    # (label, file, enclosing function, old, new)
    ("M1 revert: uninstall drops only the gains", SIM, "uninstall",
     '        if isinstance(backend_state, dict) and backend_state.get("action_controller") is self:\n'
     '            del backend_state["action_controller"]\n'
     "            deregistered = True\n",
     "        if False:  # mutated\n            pass\n"),
    ("M2 drop the identity guard (clobber any controller)", SIM, "uninstall",
     'backend_state.get("action_controller") is self',
     'backend_state.get("action_controller") is not None'),
    ("M3 from_sim stops threading the world through", SIM, "from_sim",
     "            model=model,\n            world=world,\n",
     "            model=model,\n"),
    ("M4 hook ignores the position-servo predicate", HOOK, "_maybe_install_wbc_torque_control",
     "        if not wbc_uses_position_servo(self, policy, robot_name):\n            return None\n",
     "        if False:  # mutated\n            return None\n"),
    ("M5 hook ignores an already-registered controller", HOOK, "_maybe_install_wbc_torque_control",
     '        if isinstance(backend_state, dict) and backend_state.get("action_controller") is not None:\n',
     "        if False:  # mutated\n"),
    ("M6 hook stops honouring the missing-world guard", HOOK, "_maybe_install_wbc_torque_control",
     "        if world is None or world._model is None:\n            return None\n",
     "        if False:  # mutated\n            return None\n"),
]

def run(paths, extra=()):
    cmd = [sys.executable, "-m", "pytest", *paths, *extra,
           "-q", "--no-header", "-p", "no:randomly", "--no-cov", "--tb=no"]
    env = dict(os.environ, MUJOCO_GL="egl")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)

ARM_NEW = ([NEW], ())
ARM_OLD = (["tests/policies/wbc"], (f"--ignore={NEW}",))

saved = {SIM: SIM.read_text(), HOOK: HOOK.read_text()}
rows = []
try:
    for label, path, fname, old, new in MUTATIONS:
        src = saved[path]
        lo, hi = fn_range(path, fname)
        region = "\n".join(src.splitlines()[lo - 1 : hi]) + "\n"
        in_fn, in_file = region.count(old), src.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} (in_file={in_file})"
        head = "\n".join(src.splitlines()[: lo - 1])
        tail_lines = src.splitlines()[hi:]
        mutated_region = region.replace(old, new, 1)
        mutated = (head + "\n" if head else "") + mutated_region + ("\n".join(tail_lines) + "\n" if tail_lines else "")
        assert mutated != src, f"{label}: mutation was a no-op"
        path.write_text(mutated)
        try:
            nf, npass = run(*ARM_NEW)
            of, opass = run(*ARM_OLD)
        finally:
            path.write_text(src)
        rows.append({"label": label, "in_fn": in_fn, "in_file": in_file,
                     "new_failed": nf, "new_passed": npass,
                     "old_failed": of, "old_passed": opass})
        print(f"  {label}\n      anchor in_fn={in_fn} in_file={in_file} | "
              f"new: {nf} failed/{npass} passed | pre-existing: {of} failed/{opass} passed")
    print("\n  -- unmutated control --")
    nf, npass = run(*ARM_NEW); of, opass = run(*ARM_OLD)
    rows.append({"label": "control (unmutated)", "in_fn": 0, "in_file": 0,
                 "new_failed": nf, "new_passed": npass, "old_failed": of, "old_passed": opass})
    print(f"      new: {nf} failed/{npass} passed | pre-existing: {of} failed/{opass} passed")
finally:
    for path, text in saved.items():
        path.write_text(text)

caught_new = sum(1 for r in rows if r["label"].startswith("M") and r["new_failed"] > 0)
caught_old = sum(1 for r in rows if r["label"].startswith("M") and r["old_failed"] > 0)
n_mut = sum(1 for r in rows if r["label"].startswith("M"))
print(f"\nCAUGHT by the new module: {caught_new}/{n_mut}   by the pre-existing wbc suite: {caught_old}/{n_mut}")
json.dump({"rows": rows, "caught_new": caught_new, "caught_old": caught_old, "n": n_mut},
          open(f"/tmp/mut-{RID}.json", "w"), indent=2)
