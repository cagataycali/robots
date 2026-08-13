import ast, glob, pathlib, re, shutil, subprocess, sys, json
SRC = pathlib.Path("strands_robots/mesh/transport/iot_transport.py")
NEW = "tests/mesh/test_iot_broker_delivery_routes.py"
OLD_ARM = sorted(set(glob.glob("tests/mesh/test_iot*.py") + ["tests/mesh/test_transport.py"]) - {NEW})
orig = SRC.read_text()
SAVE = pathlib.Path(f"/tmp/save-iot-{sys.argv[1]}.py"); SAVE.write_text(orig)

# (label, enclosing function or None for module level, old, new)
MUTS = [
 ("M1 put(): drop the explicit-DROP short-circuit", "put",
  "        qos, retain = _qos_and_retain_for(key)\n        if qos < 0:\n            return  # explicit DROP\n",
  "        qos, retain = _qos_and_retain_for(key)\n        if False:  # MUTATED\n            return  # explicit DROP\n"),
 ("M2 _should_drop(): match a bare kind too (make qos<0 redundant)", "_should_drop",
  "    return suffix.startswith(_NEVER_BRIDGE_PREFIXES)\n",
  "    return suffix.startswith(tuple(p.rstrip('/') for p in _NEVER_BRIDGE_PREFIXES))  # MUTATED\n"),
 ("M3 _unsubscribe(): re-raise instead of tolerating a gone handler", "_unsubscribe",
  "                except ValueError:\n                    pass  # handler already gone\n",
  "                except ValueError:\n                    raise  # MUTATED\n"),
 ("M4 _unsubscribe(): drop the missing-client guard", "_unsubscribe",
  "        if self._client is None:\n            return\n",
  "        if False:  # MUTATED\n            return\n"),
 ("M5 _is_camera_ref(): stop exempting the /ref pointer", "_is_camera_ref", None, None),
 ("M6 _TOPIC_POLICY: add a DROP entry under a top-level kind", None,
  '    "safety/estop": (1, True),\n',
  '    "safety/estop": (1, True),\n    "safety/drop-me": ("DROP", False),  # MUTATED\n'),
]

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"function {name!r} not found")

def run(paths):
    p = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly",
                        "--tb=no", "-o", "addopts="], capture_output=True, text=True)
    tail = [l for l in p.stdout.splitlines() if re.match(r"^=+.*(passed|failed|error)", l)]
    txt = tail[-1] if tail else p.stdout.strip().splitlines()[-1:]
    def g(word):
        m = re.search(rf"(\d+) {word}", str(txt));  return int(m.group(1)) if m else 0
    return g("failed") + g("error") + g("errors"), g("passed")

rows = []
try:
    f_new, p_new = run([NEW]); f_old, p_old = run(OLD_ARM)
    rows.append(("(unmutated control)", f_new, p_new, f_old, p_old))
    print(f"control: new {f_new}F/{p_new}P | pre-existing {f_old}F/{p_old}P ({len(OLD_ARM)} files)")
    for label, fn, old, new in MUTS:
        src = SAVE.read_text()
        if label.startswith("M5"):
            lo, hi = fn_range(src, fn)
            lines = src.splitlines(keepends=True)
            body = [i for i in range(lo, hi) if lines[i].lstrip().startswith("return ")]
            assert len(body) == 1, f"{label}: expected 1 return, got {len(body)}"
            lines[body[0]] = "    return False  # MUTATED\n"
            SRC.write_text("".join(lines))
        else:
            if fn:
                lo, hi = fn_range(src, fn)
                region = "".join(src.splitlines(keepends=True)[lo - 1:hi])
                in_fn, in_file = region.count(old), src.count(old)
                assert in_fn == 1, f"{label}: anchor in_fn={in_fn} in_file={in_file}"
                print(f"  {label[:3]} anchor in_fn={in_fn} in_file={in_file}")
                SRC.write_text(src.replace(region, region.replace(old, new, 1), 1))
            else:
                assert src.count(old) == 1, f"{label}: module anchor count {src.count(old)}"
                SRC.write_text(src.replace(old, new, 1))
        assert SRC.read_text() != src, f"{label}: no change applied"
        fn_, pn = run([NEW]); fo, po = run(OLD_ARM)
        rows.append((label, fn_, pn, fo, po))
        print(f"  {label:62s} new {fn_}F/{pn}P | pre-existing {fo}F/{po}P")
        SRC.write_text(src)
finally:
    shutil.copyfile(SAVE, SRC)
    assert SRC.read_text() == orig, "RESTORE FAILED"
    print("\nrestored byte-identically:", SRC.read_text() == orig)
caught = sum(1 for lbl, f, _, _, _ in rows if lbl.startswith("M") and f > 0)
blind  = sum(1 for lbl, f, _, fo, _ in rows if lbl.startswith("M") and f > 0 and fo == 0)
print(f"\ncaught by the new module: {caught}/{len(MUTS)}   invisible to the pre-existing arm: {blind}/{len(MUTS)}")
json.dump([{"label": l, "new_failed": a, "new_passed": b, "old_failed": c, "old_passed": d}
           for l, a, b, c, d in rows], open(f"/tmp/mut-{sys.argv[1]}.json", "w"), indent=2)
