import ast, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/policies/vera/provider.py")
NEW = "tests/policies/vera/test_vera_ik_bridge_lazy_build.py"
ORIG = SRC.read_text()

def fn_range(src, name):
    t = ast.parse(src)
    for f in ast.walk(t):
        if isinstance(f, ast.FunctionDef) and f.name == name:
            return f.lineno, f.end_lineno
    raise AssertionError(name)

MUTS = [
    ("M1 drop the cache guard (rebuild every inference)",
     "        if self._ik_bridge is None:\n            from .sim_ik import MinkIKBridge\n\n            self._ik_bridge = MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)\n",
     "        from .sim_ik import MinkIKBridge\n\n        self._ik_bridge = MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)\n"),
    ("M2 swap the frame name and type arguments",
     "MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)",
     "MinkIKBridge(mj_model, self._ee_frame_type, ee_frame)"),
    ("M3 hardcode the frame type as \"body\"",
     "MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)",
     'MinkIKBridge(mj_model, ee_frame, "body")'),
    ("M4 build without caching (return a fresh bridge)",
     "            self._ik_bridge = MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)\n        return self._ik_bridge\n",
     "            return MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)\n        return self._ik_bridge\n"),
]

def run(target):
    r = subprocess.run([sys.executable, "-m", "pytest", *target.split(), "-q", "--no-header",
                        "-p", "no:randomly", "--no-cov", "--tb=no"], capture_output=True, text=True)
    out = r.stdout
    f = re.search(r"(\d+) failed", out); p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)

lo, hi = fn_range(ORIG, "_ensure_ik_bridge")
print(f"_ensure_ik_bridge spans lines {lo}..{hi}\n")
ARM_B = f"tests/policies/vera --ignore={NEW}"
print(f"{'mutation':<50} {'new file':>14}  {'pre-existing vera':>18}")
print("-" * 88)
try:
    for label, old, new in MUTS:
        region = "\n".join(ORIG.splitlines()[lo-1:hi]) + "\n"
        in_fn, in_file = region.count(old), ORIG.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside the function"
        SRC.write_text(ORIG.replace(old, new, 1))
        ast.parse(SRC.read_text())
        a, b = run(NEW), run(ARM_B)
        print(f"{label:<50} {a[0]:>3} failed/{a[1]:>3} pass  {b[0]:>3} failed/{b[1]:>4} pass"
              f"   [anchor in_fn={in_fn} in_file={in_file}]")
        SRC.write_text(ORIG)
    print("-" * 88)
    a, b = run(NEW), run(ARM_B)
    print(f"{'(unmutated control)':<50} {a[0]:>3} failed/{a[1]:>3} pass  {b[0]:>3} failed/{b[1]:>4} pass")
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG
    print("\nrestored byte-identically:", SRC.read_text() == ORIG)
