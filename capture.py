"""Re-derive every number the figure shows, from the tree it runs in."""
import ast, json, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
assert pathlib.Path(strands_robots.__file__).parents[1] == ROOT, "wrong tree on sys.path"

SEC = ROOT / "strands_robots/mesh/security.py"
ACL = ROOT / "strands_robots/mesh/_acl_config.py"
TESTS = ["tests/mesh/test_policy_host_charset_gate.py",
         "tests/mesh/test_acl_config.py",
         "tests/mesh/test_validate_command_finite_numerics.py"]
BASE = subprocess.run(["git", "rev-parse", "upstream/main"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip()

def fn_range(path, name):
    src = path.read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return src, n.lineno, n.end_lineno
    raise AssertionError(name)

def mutate(path, fn, old, new):
    src, lo, hi = fn_range(path, fn)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    assert region.count(old) == 1, (fn, region.count(old))
    out = "".join(lines[:lo - 1]) + region.replace(old, new, 1) + "".join(lines[hi:])
    ast.parse(out)
    path.write_text(out)

def run():
    r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:randomly",
                        "--no-cov", "--timeout=120", "--tb=no"], cwd=ROOT,
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l][-1]
    failed = 0
    for tok in line.replace("=", " ").split():
        pass
    import re
    m = re.search(r"(\d+) failed", line)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", line)
    passed = int(m.group(1)) if m else 0
    return {"failed": failed, "passed": passed}

POST_CHECK = '''        host_str = str(policy_host)
        if not _SAFE_PASSTHROUGH_RE.fullmatch(host_str):
            raise ValidationError(
                f"policy_host={policy_host!r} contains control characters (CRLF/NUL/C0). Use printable ASCII only."
            )
'''
FINITE_GUARD = '''    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
'''
STATIC = '''    if path.is_symlink():
        raise ValueError(
            f"refusing to load ACL file {path}: it is a SYMLINK "
            f"(target: {os.readlink(path)!r}). ACL files must be regular files."
        )
'''
MUTATIONS = [
    ("M1", "delete the policy_host post-check", SEC, "validate_command",
     POST_CHECK, "        host_str = str(policy_host)\n"),
    ("M2", "drop O_NOFOLLOW from the open flags", ACL, "_load_acl_file",
     '    nofollow = getattr(os, "O_NOFOLLOW", 0)\n', "    nofollow = 0\n"),
    ("M3", "narrow the ELOOP handler so the race escapes", ACL, "_load_acl_file",
     "    except OSError as exc:\n", "    except PermissionError as exc:\n"),
    ("M4", "delete _coerce_int's finite guard", SEC, "_coerce_int", FINITE_GUARD, ""),
    ("M5", "delete the static is_symlink check", ACL, "_load_acl_file", STATIC, ""),
]

save = pathlib.Path(tempfile.mkdtemp())
for t in TESTS:
    shutil.copy2(ROOT / t, save / pathlib.Path(t).name)
orig = {SEC: SEC.read_text(), ACL: ACL.read_text()}

out = {"tree": str(ROOT), "base": BASE, "mutations": []}
out["clean"] = {"pr": run()}
subprocess.run(["git", "checkout", BASE, "--", *TESTS], cwd=ROOT, check=True)
out["clean"]["main"] = run()
for t in TESTS:
    shutil.copy2(save / pathlib.Path(t).name, ROOT / t)

for tag, label, path, fn, old, new in MUTATIONS:
    mutate(path, fn, old, new)
    pr = run()
    subprocess.run(["git", "checkout", BASE, "--", *TESTS], cwd=ROOT, check=True)
    mn = run()
    for t in TESTS:
        shutil.copy2(save / pathlib.Path(t).name, ROOT / t)
    for p, s in orig.items():
        p.write_text(s)
    out["mutations"].append({"tag": tag, "label": label, "pr": pr, "main": mn})
    print(f"  {tag} {label:44s} PR={pr}  main={mn}")

subprocess.run(["git", "reset", "-q", "HEAD", "--", *TESTS], cwd=ROOT, check=False)
for p, s in orig.items():
    assert p.read_text() == s, f"{p} not restored"
for t in TESTS:
    assert (ROOT / t).read_text() == (save / pathlib.Path(t).name).read_text(), t
print("  restore verified byte-identical")

# coverage accounting, read from the two full-suite runs
cb = json.load(open(sys.argv[1]))
ca = json.load(open(sys.argv[2]))
out["coverage"] = {}
for p in ("strands_robots/mesh/security.py", "strands_robots/mesh/_acl_config.py"):
    b, a = cb["files"][p], ca["files"][p]
    out["coverage"][p] = {
        "main_missing": sorted(b["missing_lines"]),
        "pr_missing": sorted(a["missing_lines"]),
        "main_pct": b["summary"]["percent_covered"],
        "pr_pct": a["summary"]["percent_covered"],
    }
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
print("wrote", sys.argv[3])
