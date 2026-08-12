"""Mutation table: 2 styles x 2 arms. Arm B is upstream's own copy of the test file."""
import ast, json, os, pathlib, re, shutil, subprocess, sys

SRC = pathlib.Path("strands_robots/mesh/session.py")
TEST = pathlib.Path("tests/mesh/test_zenoh_lifecycle_error_classification.py")
BASE_TEST = pathlib.Path("tests/mesh/test_zz_base_probe.py")
RUN = os.environ["GITHUB_RUN_ID"]

orig_src = SRC.read_text()
shutil.copy(SRC, f"/tmp/save-session-{RUN}.py")
# Arm B: upstream's copy of the test module, dropped in so it collects.
base_test = subprocess.run(["git", "show", f"6e905fab:{TEST}"], capture_output=True, text=True).stdout
BASE_TEST.write_text(base_test)

def fnrange(src, name):
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return fn.lineno, fn.end_lineno

def mutate(name, old, new):
    """Replace `old` with `new` inside function `name` only; print in_fn vs in_file."""
    src = orig_src
    lo, hi = fnrange(src, name)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{name}: anchor x{in_fn} in fn (x{in_file} in file)"
    lines[lo - 1 : hi] = [region.replace(old, new, 1)]
    out = "".join(lines)
    assert out != src
    ast.parse(out)
    SRC.write_text(out)
    return in_fn, in_file

def run(paths):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"],
                       capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    m = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return (int(m.group(1)) if m else 0), (int(p.group(1)) if p else 0)

BARE = "            except Exception:\n                pass\n"
MUTS = [
    ("M1 release_session: revert to a bare swallow", "release_session",
     '''            except zenoh_error_types() as exc:''', '''            except Exception as exc:  # noqa: BLE001
                del exc'''),
    ("M2 release_session: keep the narrow catch, drop the record", "release_session",
     '''                logger.warning("Zenoh mesh session close failed: %s", exc)''',
     '''                del exc'''),
    ("M3 release_session: report the close unconditionally again", "release_session",
     '''            if closed:\n                logger.info''', '''            if True:  # noqa: SIM103\n                logger.info'''),
    ("M4 _atexit_cleanup: revert to a bare swallow", "_atexit_cleanup",
     '''            except zenoh_error_types() as exc:''', '''            except Exception as exc:  # noqa: BLE001
                del exc'''),
    ("M5 _atexit_cleanup: keep the narrow catch, drop the record", "_atexit_cleanup",
     '''                logger.debug("Zenoh mesh session close failed at exit: %s", exc)''',
     '''                del exc'''),
    ("M6 release_session: downgrade the record to DEBUG", "release_session",
     '''                logger.warning("Zenoh mesh session close failed''',
     '''                logger.debug("Zenoh mesh session close failed'''),
]

rows = []
try:
    unm_a, unm_b = run([str(TEST)]), run([str(BASE_TEST)])
    rows.append(("(unmutated control)", unm_a, unm_b, "-"))
    for label, fn, old, new in MUTS:
        counts = mutate(fn, old, new)
        a, b = run([str(TEST)]), run([str(BASE_TEST)])
        rows.append((label, a, b, f"in_fn={counts[0]} in_file={counts[1]}"))
        SRC.write_text(orig_src)
finally:
    shutil.copy(f"/tmp/save-session-{RUN}.py", SRC)
    BASE_TEST.unlink(missing_ok=True)
    assert SRC.read_text() == orig_src, "source not restored byte-identically"

print(f"{'mutation':<52} {'this PR':>14} {'upstream tests':>16}  anchor")
for label, (fa, pa), (fb, pb), anc in rows:
    blind = "  <- BLIND" if fb == 0 and label.startswith("M") else ""
    print(f"{label:<52} {f'{fa} failed':>14} {f'{fb} failed':>16}  {anc}{blind}")
caught_new = sum(1 for l, (fa, _), _, _ in rows if l.startswith("M") and fa > 0)
caught_old = sum(1 for l, _, (fb, _), _ in rows if l.startswith("M") and fb > 0)
n = sum(1 for l, *_ in rows if l.startswith("M"))
print(f"\ncaught by this PR: {caught_new} of {n}   caught by the pre-existing tests: {caught_old} of {n}")
print(f"restored byte-identically: {SRC.read_text() == orig_src}")
json.dump({"rows": [[l, a, b] for l, a, b, _ in rows], "caught_new": caught_new,
           "caught_old": caught_old, "n": n},
          open(f"/tmp/mutate-{RUN}.json", "w"), indent=1)
