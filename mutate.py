import ast, json, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/tools/use_rosbridge.py")
NEW = pathlib.Path("tests/tools/test_use_rosbridge.py")
OLD = pathlib.Path("tests/tools/test_zz_main_baseline_probe.py")

# arm B: main's copy of the test file, dropped in so it collects
OLD.write_text(
    subprocess.run(["git", "show", "upstream/main:tests/tools/test_use_rosbridge.py"],
                   capture_output=True, text=True, check=True).stdout, encoding="utf-8")

src0 = SRC.read_text(encoding="utf-8")
new0 = NEW.read_text(encoding="utf-8")

fn = next(n for n in ast.walk(ast.parse(src0))
          if isinstance(n, ast.FunctionDef) and n.name == "connect")
lo, hi = fn.lineno, fn.end_lineno
region = "".join(src0.splitlines(keepends=True)[lo - 1:hi])

PROD = [
    ("M1 delete the ready-without-connection guard",
     '''            if not getattr(ros, "is_connected", False):
                raise TimeoutError(
                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    "- is rosbridge_server running?"
                )
''', ""),
    ("M2 give it the raised-dial wording (indistinguishable)",
     '''                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    "- is rosbridge_server running?"
                )
            return ros''',
     '''                    f"could not connect to rosbridge at ws://{host}:{port} within {timeout}s "
                    "- is rosbridge_server running? (connection refused)"
                )
            return ros'''),
    ("M3 wait loop never returns early",
     """            if getattr(ros, "is_connected", False):
                return ros
            time.sleep(0.05)""",
     """            if getattr(ros, "is_connected", False):
                pass
            time.sleep(0.05)"""),
    ("M4 do not cache a not-yet-connected entry",
     """            ros = roslibpy.Ros(host=host, port=port)
            self._connections[(host, port)] = ros
            try:""",
     """            ros = roslibpy.Ros(host=host, port=port)
            try:"""),
]

def run(paths: list[str]) -> tuple[int, int]:
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-header",
                        "-p", "no:randomly", "--no-cov"], capture_output=True, text=True)
    out = r.stdout
    return out.count("FAILED"), sum(int(w) for l in out.splitlines() if " passed" in l
                                    for w in l.split() if w.isdigit() and " passed" in l) or -1

def counts(paths: list[str]) -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-header",
                        "-p", "no:randomly", "--no-cov"], capture_output=True, text=True)
    f = r.stdout.count("FAILED")
    m = re.search(r"(\d+) passed", r.stdout)
    return f, (int(m.group(1)) if m else -1)

rows = []
try:
    for label, old, new in PROD:
        in_fn, in_file = region.count(old), src0.count(old)
        assert in_fn == 1 and in_file == 1, (label, in_fn, in_file)
        print(f"{label}: anchor in_fn={in_fn} in_file={in_file}")
        SRC.write_text(src0.replace(old, new, 1), encoding="utf-8")
        assert ast.parse(SRC.read_text())
        fa, sa = counts([str(NEW)])
        fb, sb = counts([str(OLD)])
        rows.append((label, fa, sa, fb, sb))
        SRC.write_text(src0, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == src0

    # M5 is test-side: revert __set__ so _Flapping is a non-data descriptor again
    m5_old = '''        def __set__(self, obj: Any, value: bool) -> None:
            """Absorb writes so the scripted reads decide, not the instance dict."""

'''
    assert new0.count(m5_old) == 1
    NEW.write_text(new0.replace(m5_old, "", 1), encoding="utf-8")
    fa, sa = counts([str(NEW)])
    rows.append(("M5 revert __set__ (non-data descriptor again)", fa, sa, "n/a", "n/a"))
finally:
    SRC.write_text(src0, encoding="utf-8")
    NEW.write_text(new0, encoding="utf-8")
    OLD.unlink(missing_ok=True)

assert SRC.read_text(encoding="utf-8") == src0 and NEW.read_text(encoding="utf-8") == new0
print("\nsources restored byte-identically\n")
hdr_a, hdr_b = "this PR", "main's copy"
print(f"{'mutation':<46} {hdr_a:<26} {hdr_b:<26}")
print("-" * 100)
for label, fa, sa, fb, sb in rows:
    a = f"{fa} failed / {sa} passed"
    b = "n/a (test-side)" if fb == "n/a" else f"{fb} failed / {sb} passed"
    print(f"{label:<48} {a:<24} {b}")
json.dump([{"m": r[0], "new_failed": r[1], "new": r[2], "old_failed": r[3], "old": r[4]} for r in rows],
          open("/tmp/mut.json", "w"), indent=1)
