import ast, os, pathlib, re, subprocess, sys

MINE = pathlib.Path(open("/tmp/minepath-" + os.environ["GITHUB_RUN_ID"]).read().strip())
BASE = MINE / "strands_robots/simulation/base.py"
MUJ = MINE / "strands_robots/simulation/mujoco/simulation.py"
FAC = MINE / "strands_robots/policies/factory.py"

NEW = ["tests/simulation/test_unresolvable_policy_provider.py"]
OLD = [
    "tests/simulation/test_eval_facade_preflight_refusals.py",
    "tests/simulation/test_policy_preflight_fail_fast.py",
    "tests/simulation/test_policy_config_mapping_validation.py",
    "tests/simulation/test_run_policy_horizon_validation.py",
    "tests/simulation/test_policy_kwargs_forwarding.py",
    "tests/policies/test_factory.py",
]

PROBE = """        reason = policy_provider_error(policy_provider, **(policy_config or {}))
        if reason is not None:
            return {"status": "error", "content": [{"text": reason}]}
"""
GUARD = """        if err := self._unresolvable_policy_provider_error(policy_provider, policy_config):
            return err
"""

MUTATIONS = [
    ("M1 drop the base-facade probe", BASE, "_preflight_policy_config", PROBE, ""),
    ("M2 probe, discard the reason", BASE, "_preflight_policy_config", PROBE,
     "        policy_provider_error(policy_provider, **(policy_config or {}))\n"),
    ("M3 drop the start_policy guard", MUJ, "start_policy", GUARD, ""),
    ("M4 guard, discard the envelope", MUJ, "start_policy", GUARD,
     "        self._unresolvable_policy_provider_error(policy_provider, policy_config)\n"),
    ("M5 reword the reason locally", BASE, "_preflight_policy_config",
     '            return {"status": "error", "content": [{"text": reason}]}\n',
     '            return {"status": "error", "content": [{"text": "bad provider"}]}\n'),
    ("M6 widen the probe to Exception", FAC, "policy_provider_error",
     "    except ValueError as e:\n", "    except Exception as e:\n"),
]


def fn_range(path, name):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"{name} not in {path.name}")


def run(files):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
        cwd=MINE, capture_output=True, text=True,
        env={**os.environ, "MUJOCO_GL": "egl"}, timeout=1200,
    )
    out = r.stdout + r.stderr
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    return f, p


rows = []
for label, path, fname, old, new in MUTATIONS:
    src = path.read_text()
    lo, hi = fn_range(path, fname)
    region = "".join(src.splitlines(keepends=True)[lo - 1 : hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{label}: in_fn={in_fn} (in_file={in_file})"
    print(f"{label}: anchor in_fn={in_fn} in_file={in_file}")
    mutated = src.replace(region, region.replace(old, new, 1), 1)
    assert mutated != src
    ast.parse(mutated)
    try:
        path.write_text(mutated)
        nf, np_ = run(NEW)
        of, op = run(OLD)
        rows.append((label, nf, np_, of, op))
        print(f"   new: {nf} failed/{np_} passed   pre-existing: {of} failed/{op} passed")
    finally:
        path.write_text(src)
        assert path.read_text() == src

print("\nunmutated control:")
print("   new:", run(NEW), "  pre-existing:", run(OLD))
caught_new = sum(1 for r in rows if r[1] > 0)
caught_old = sum(1 for r in rows if r[3] > 0)
print(f"\ncaught by the new module: {caught_new}/{len(rows)}   by the pre-existing suite: {caught_old}/{len(rows)}")
import json
json.dump([{"label": r[0], "new_failed": r[1], "old_failed": r[3]} for r in rows],
          open(f"/tmp/pw-{os.environ['GITHUB_RUN_ID']}/mut.json", "w"), indent=1)
