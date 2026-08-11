import ast, pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path.cwd()
SRC = ROOT / "strands_robots/simulation/newton/randomization.py"
NEW_FILE = "tests/simulation/newton/test_domain_randomization.py"
MJ_FILE = "tests/simulation/mujoco/test_randomization_option_guards.py"
NEW_K = "TestSeedRefusalOnBothEntryPoints"

GUARD_R = ('        if msg := randomization_seed_error(seed, "randomize"):\n'
           '            return {"status": "error", "content": [{"text": msg}]}\n')
GUARD_N = ('        if msg := randomization_seed_error(seed, "set_obs_noise"):\n'
           '            return {"status": "error", "content": [{"text": msg}]}\n')

MUTATIONS = [
    ("M1 randomize: delete the seed guard", "randomize", GUARD_R, ""),
    ("M2 randomize: call the guard, discard the verdict", "randomize", GUARD_R,
     '        randomization_seed_error(seed, "randomize")\n'),
    ("M3 set_obs_noise: delete the seed guard", "set_obs_noise", GUARD_N, ""),
    ("M4 set_obs_noise: call the guard, discard the verdict", "set_obs_noise", GUARD_N,
     '        randomization_seed_error(seed, "set_obs_noise")\n'),
    ("M5 randomize: reword the reason locally", "randomize", GUARD_R,
     '        if randomization_seed_error(seed, "randomize"):\n'
     '            return {"status": "error", "content": [{"text": "bad seed"}]}\n'),
    ("M6 randomize: copy the rollout ceiling onto this path", "randomize", GUARD_R,
     '        if msg := randomization_seed_error(seed, "randomize", max_seed=4294967295):\n'
     '            return {"status": "error", "content": [{"text": msg}]}\n'),
    ("M7 set_obs_noise: copy the rollout ceiling onto this path", "set_obs_noise", GUARD_N,
     '        if msg := randomization_seed_error(seed, "set_obs_noise", max_seed=4294967295):\n'
     '            return {"status": "error", "content": [{"text": msg}]}\n'),
]

def fn_range(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)

def run(paths, k=None):
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--no-header",
           "-p", "no:randomly", "--no-cov", "--tb=no"]
    if k:
        cmd += ["-k", k]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"}).stdout
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

original = SRC.read_text()
backup = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
backup.write(original); backup.close()

print(f"{'mutation':52} {'new class':>14} {'pre-existing':>14}")
print("-" * 84)
try:
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(original, fname)
        region = "".join(original.splitlines(keepends=True)[lo - 1:hi])
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname}()"
        # rewrite only inside the function's own line range
        lines = original.splitlines(keepends=True)
        head, body, tail = lines[:lo - 1], region, lines[hi:]
        mutated = "".join(head) + body.replace(old, new, 1) + "".join(tail)
        assert mutated != original
        ast.parse(mutated)
        SRC.write_text(mutated)
        a_f, a_p = run([NEW_FILE], NEW_K)
        b_f, b_p = run([NEW_FILE, MJ_FILE], f"not {NEW_K}")
        SRC.write_text(original)
        a = f"{a_f} failed" if a_f else f"BLIND ({a_p} pass)"
        b = f"{b_f} failed" if b_f else f"BLIND ({b_p} pass)"
        print(f"{label:52} {a:>14} {b:>14}   [in_fn={in_fn} in_file={in_file}]")
    # control
    a_f, a_p = run([NEW_FILE], NEW_K)
    b_f, b_p = run([NEW_FILE, MJ_FILE], f"not {NEW_K}")
    print(f"{'(unmutated control)':52} {str(a_p)+' pass':>14} {str(b_p)+' pass':>14}")
finally:
    SRC.write_text(original)
    same = SRC.read_text() == pathlib.Path(backup.name).read_text()
    print(f"\nsource restored byte-identical: {same}")
    assert same
