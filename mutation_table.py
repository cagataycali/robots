"""Mutation table: 6 plausible regressions x 2 arms (new tests / pre-existing)."""
import ast, json, os, pathlib, subprocess

MINE = pathlib.Path("/tmp/robots-mine-" + os.environ["GITHUB_RUN_ID"])
PY = MINE / "strands_robots" / "registry" / "policies.py"
JS = MINE / "strands_robots" / "registry" / "policies.json"
NEW = "tests/registry/test_provider_import_error_names_its_remedy.py"

ORIG_PY = PY.read_text(encoding="utf-8")
ORIG_JS = JS.read_text(encoding="utf-8")


def fn_range(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)


def scoped_replace(src, fname, old, new):
    """Replace `old` exactly once INSIDE function `fname`; print in_fn vs in_file."""
    lo, hi = fn_range(src, fname)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    in_fn, in_file = region.count(old), src.count(old)
    print(f"      anchor in_fn={in_fn} in_file={in_file}")
    assert in_fn == 1, f"anchor not unique in {fname}: {in_fn}"
    new_region = region.replace(old, new)
    return "".join(lines[: lo - 1]) + new_region + "".join(lines[hi:])


MUTATIONS = [
    ("M1 registered branch: no translation (raw ImportError escapes)", "py", "import_policy_class",
     '''        try:
            mod = importlib.import_module(config["module"])
        except ImportError as exc:''',
     '''        if True:
            mod = importlib.import_module(config["module"])
        if False:
            exc = None'''),
    ("M2 keep the call, discard the translation (re-raise the original)", "py", "import_policy_class",
     "            raise _provider_import_error(canonical, exc, config.get(\"extra\")) from exc",
     "            _provider_import_error(canonical, exc, config.get(\"extra\"))\n            raise"),
    ("M3 auto-discovery: swallow the ImportError again", "py", "import_policy_class",
     '''        if getattr(exc, "name", None) != f"strands_robots.policies.{provider}":
            raise _provider_import_error(provider, exc, None) from exc''',
     "        pass"),
    ("M4 ignore the declared extra (remedy loses the install command)", "py", "_provider_import_error",
     "    if extra:", "    if False:"),
    ("M5 substitute the mock provider on a failed import", "py", "import_policy_class",
     "            raise _provider_import_error(canonical, exc, config.get(\"extra\")) from exc",
     '''            from strands_robots.policies.mock import MockPolicy
            return MockPolicy'''),
    ("M6 drop lerobot_local's extra from the registry", "js", None,
     '      "extra": "lerobot",\n', ""),
]


def run(target):
    r = subprocess.run(
        ["python3", "-m", "pytest", *target, "-q", "--no-cov", "-p", "no:randomly"],
        cwd=MINE, capture_output=True, text=True, timeout=900)
    out = r.stdout
    import re
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)


PRE = ["tests/registry", f"--ignore={NEW}"]
print("baseline (unmutated):")
print("   new tests      :", run([NEW]))
print("   pre-existing   :", run(PRE))
print()
rows = []
try:
    for label, kind, fname, old, new in MUTATIONS:
        print(f"-> {label}")
        if kind == "py":
            PY.write_text(scoped_replace(ORIG_PY, fname, old, new), encoding="utf-8")
            ast.parse(PY.read_text(encoding="utf-8"))
        else:
            s = ORIG_JS
            assert s.count(old) == 1, "json anchor"
            JS.write_text(s.replace(old, new), encoding="utf-8")
            json.loads(JS.read_text(encoding="utf-8"))
        nf, npass = run([NEW])
        pf, ppass = run(PRE)
        rows.append((label, nf, pf))
        print(f"      new tests: {nf} failed / {npass} passed | pre-existing: {pf} failed / {ppass} passed")
        PY.write_text(ORIG_PY, encoding="utf-8"); JS.write_text(ORIG_JS, encoding="utf-8")
finally:
    PY.write_text(ORIG_PY, encoding="utf-8"); JS.write_text(ORIG_JS, encoding="utf-8")
    assert PY.read_text(encoding="utf-8") == ORIG_PY and JS.read_text(encoding="utf-8") == ORIG_JS
    print("\nrestored byte-identically")

print("\n| mutation | new tests | pre-existing |")
for label, nf, pf in rows:
    print(f"| {label} | {nf} failed | {pf} failed |")
caught = sum(1 for _l, nf, _p in rows if nf > 0)
blind = sum(1 for _l, nf, pf in rows if nf > 0 and pf == 0)
print(f"\ncaught by new: {caught}/{len(rows)} | invisible to pre-existing: {blind}/{len(rows)}")
pathlib.Path(f"/tmp/mutation-{os.environ['GITHUB_RUN_ID']}.json").write_text(json.dumps(rows, indent=2))
