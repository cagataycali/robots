import ast, pathlib, re, subprocess, sys

NEWTON = pathlib.Path("strands_robots/simulation/newton/simulation.py")
ISAAC = pathlib.Path("strands_robots/simulation/isaac/simulation.py")
TESTS = pathlib.Path("tests/simulation/test_timestep_domain_across_surfaces.py")
BASE = subprocess.run(["git", "merge-base", "HEAD", "upstream/main"], capture_output=True, text=True).stdout.strip()
PROBE = pathlib.Path("tests/simulation/test_zz_base_probe.py")

GUARD = ('        if err := self._validate_timestep(effective_timestep, "create_world", timestep_param):\n'
         "            return err\n")

def fn_range(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for m in ast.iter_child_nodes(node):
                if isinstance(m, ast.FunctionDef) and m.name == name:
                    return m.lineno, m.end_lineno
    raise SystemExit(f"no {name} in {path}")

def scoped(path, name, old, new):
    """Replace `old`->`new` only inside `name`, printing in_fn vs in_file."""
    lo, hi = fn_range(path, name)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    whole = "".join(lines)
    in_fn, in_file = region.count(old), whole.count(old)
    print(f"      anchor in_fn={in_fn} in_file={in_file}  ({path.name}::{name})")
    assert in_fn == 1, f"in_fn={in_fn}"
    return whole[: len("".join(lines[: lo - 1]))] + region.replace(old, new, 1) + "".join(lines[hi:])

ARG_ONLY = ('        if timestep is not None and (err := self._validate_timestep(timestep, "create_world", "timestep")):\n'
            "            return err\n")
INLINE = ("        import math as _m\n"
          "        if not isinstance(effective_timestep, (int, float)) or not _m.isfinite(effective_timestep) or effective_timestep <= 0:\n"
          '            return {"status": "error", "content": [{"text": "bad timestep"}]}\n')

MUTATIONS = [
    ("M1 newton: keep the call, discard the refusal", NEWTON, "create_world", GUARD, GUARD.replace("return err", "pass")),
    ("M2 newton: delete the timestep guard",          NEWTON, "create_world", GUARD, "        pass\n"),
    ("M3 isaac:  keep the call, discard the refusal", ISAAC,  "create_world", GUARD, GUARD.replace("return err", "pass")),
    ("M4 isaac:  delete the timestep guard",          ISAAC,  "create_world", GUARD, "        pass\n"),
    ("M5 newton: validate the argument, not the effective dt", NEWTON, "create_world", GUARD, ARG_ONLY),
    ("M6 isaac:  validate the argument, not the effective dt", ISAAC,  "create_world", GUARD, ARG_ONLY),
    ("M7 isaac:  blame `timestep` for a bad engine default", ISAAC, "create_world",
     '        timestep_param = "physics_dt" if timestep is None else "timestep"\n',
     '        timestep_param = "timestep"\n'),
    ("M8 newton: hand-roll the domain in create_world", NEWTON, "create_world", GUARD, INLINE),
]

NEW_CLASSES = ("TestEveryWorldBuilderRefusesWhatNoIntegratorCanHonor or "
               "TestAnUnusableEngineDefaultIsNamedUnderItsOwnKnob or "
               "TestTheConfigGuardCannotSeeEveryUnusableDefault or "
               "TestARefusedWorldBuilderCostsNoSolverWork or "
               "TestTheEngineDefaultSentinelIsArgumentOnly or "
               "TestNoBackendCanShipAnUnsharedTimestepDomain")

def run(target, k=None):
    cmd = [sys.executable, "-m", "pytest", str(target), "-q", "--no-cov", "-p", "no:randomly", "--tb=no"]
    if k:
        cmd += ["-k", k]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if re.match(r"^=+.*(passed|failed|error)", l)]
    tail = lines[-1] if lines else out.strip().splitlines()[-1:]
    f = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", str(tail)))
    p = int(re.search(r"(\d+) passed", str(tail)).group(1)) if re.search(r"(\d+) passed", str(tail)) else 0
    return f, p

saved = {p: p.read_text(encoding="utf-8") for p in (NEWTON, ISAAC)}
PROBE.write_text(subprocess.run(["git", "show", f"{BASE}:{TESTS}"], capture_output=True, text=True).stdout, encoding="utf-8")
try:
    print("=== unmutated control ===")
    print("   new cells:", run(TESTS, NEW_CLASSES), "  base-version file:", run(PROBE))
    print()
    for label, path, fname, old, new in MUTATIONS:
        print(f"--- {label}")
        path.write_text(scoped(path, fname, old, new), encoding="utf-8")
        try:
            a = run(TESTS, NEW_CLASSES)
            b = run(PROBE)
            print(f"      new cells: {a[0]:>2} failed / {a[1]:>3} passed   |   base version: {b[0]:>2} failed / {b[1]:>3} passed")
        finally:
            path.write_text(saved[path], encoding="utf-8")
finally:
    for p, text in saved.items():
        p.write_text(text, encoding="utf-8")
    PROBE.unlink(missing_ok=True)
    for p in (NEWTON, ISAAC):
        assert p.read_text(encoding="utf-8") == saved[p], f"{p} not restored"
    print("\nrestored byte-identically; probe removed")
