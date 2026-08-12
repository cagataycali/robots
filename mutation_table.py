import ast, json, os, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/hardware_robot.py")
ORIG = SRC.read_text()
FN = "start_teleop_receive"
NEWCLS = "TestAnAcceptedCallReplacesTheLiveStream"
FILE = "tests/mesh/test_teleop_identifier_source_scoping.py"
PUB  = "tests/test_teleop_rate_and_duration_guards.py"

tree = ast.parse(ORIG)
lo = hi = None
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name == FN:
        lo, hi = n.lineno, n.end_lineno
assert lo, FN
region = "\n".join(ORIG.splitlines()[lo-1:hi]) + "\n"

TEARDOWN = '''        key = f"{source_peer_id}/{device_name}"
        if key in self._input_receivers:
            self._input_receivers[key].stop()
'''
VALIDATE = '''        try:
            validate_mesh_identifier(source_peer_id, "start_teleop_receive.source_peer_id")
            validate_mesh_identifier(device_name, "start_teleop_receive.device_name")
        except ValidationError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
'''
REGISTER = '''        self._input_receivers[key] = receiver
'''

MUTS = [
    ("M1 delete the teardown entirely", TEARDOWN,
     '        key = f"{source_peer_id}/{device_name}"\n'),
    ("M2 keep the lookup, drop the .stop()", TEARDOWN,
     '        key = f"{source_peer_id}/{device_name}"\n'
     '        if key in self._input_receivers:\n            pass\n'),
    ("M3 tear down BEFORE validating", VALIDATE + "\n        from strands_robots.mesh import InputReceiver\n",
     '        _k = f"{source_peer_id}/{device_name}"\n'
     '        if getattr(self, "_input_receivers", {}).get(_k) is not None:\n'
     '            self._input_receivers[_k].stop()\n' + VALIDATE +
     "\n        from strands_robots.mesh import InputReceiver\n"),
    ("M4 key on device_name alone", TEARDOWN,
     '        key = f"{source_peer_id}/{device_name}"\n'
     '        if device_name in self._input_receivers:\n'
     '            self._input_receivers[device_name].stop()\n'),
    ("M5 never register the replacement", REGISTER, "        pass  # dropped\n"),
]

def run(paths, kexpr=None):
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "--no-cov", "-p", "no:randomly"]
    if kexpr: cmd += ["-k", kexpr]
    env = dict(os.environ, MUJOCO_GL="egl")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    p = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    return f, p

print(f"{'mutation':40s} | {'new class':>14s} | {'pre-existing':>14s}")
print("-" * 76)
f, p = run([FILE], NEWCLS); f2, p2 = run([FILE, PUB], f"not {NEWCLS}")
print(f"{'(unmutated control)':40s} | {f} failed/{p:3d} pass | {f2} failed/{p2:3d} pass")
rows = []
try:
    for label, old, new in MUTS:
        in_fn, in_file = region.count(old), ORIG.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        mutated = ORIG.replace(old, new, 1)
        assert mutated != ORIG, label
        SRC.write_text(mutated)
        a = run([FILE], NEWCLS)
        b = run([FILE, PUB], f"not {NEWCLS}")
        rows.append((label, in_fn, in_file, a, b))
        print(f"{label:40s} | {a[0]} failed/{a[1]:3d} pass | {b[0]} failed/{b[1]:3d} pass"
              f"   [anchor in_fn={in_fn} in_file={in_file}]")
finally:
    SRC.write_text(ORIG)
assert SRC.read_text() == ORIG, "restore failed"
print("\nrestore: byte-identical OK")
blind = [r[0] for r in rows if r[4][0] == 0]
print(f"caught by the new class: {sum(1 for r in rows if r[3][0] > 0)}/{len(rows)}")
print(f"INVISIBLE to the pre-existing tests: {len(blind)}/{len(rows)} -> {blind}")
json.dump({"rows": [{"label": r[0], "new": r[3], "old": r[4]} for r in rows]},
          open(f"/tmp/mut-{os.environ['GITHUB_RUN_ID']}.json", "w"), indent=1)
