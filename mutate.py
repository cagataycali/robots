"""Mutation table: does anything catch a regression in the unavailability block?"""
import ast, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
print("TREE:", ROOT)
NEW = ["tests/simulation/test_recording_dataset_stack_unavailable_across_backends.py"]
OLD = [
    "tests/simulation/mujoco/test_recording_backends.py",
    "tests/simulation/mujoco/test_recording_paths.py",
    "tests/simulation/newton/test_recording_lifecycle_guards.py",
    "tests/simulation/newton/test_dataset_recording.py",
    "tests/simulation/isaac/test_dataset_recording.py",
    "tests/simulation/test_recording_preflight_refusals_across_backends.py",
    "tests/simulation/test_recording_posture_flag_domain.py",
    "tests/simulation/test_dataset_recording_fps_contract.py",
]

def fn_range(path, name):
    tree = ast.parse((ROOT/path).read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"{name} not in {path}")

def apply(path, fname, old, new):
    p = ROOT/path
    src = p.read_text()
    lines = src.splitlines(keepends=True)
    lo, hi = fn_range(path, fname)
    region = "".join(lines[lo-1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"anchor in_fn={in_fn} (in_file={in_file}) for {path}::{fname}"
    print(f"      anchor in_fn={in_fn} in_file={in_file}  [{path}::{fname}]")
    region2 = region.replace(old, new, 1)
    out = "".join(lines[:lo-1]) + region2 + "".join(lines[hi:])
    assert out != src
    ast.parse(out)
    p.write_text(out)
    return src

def run(files):
    r = subprocess.run([sys.executable, "-m", "pytest", *files, "-q", "-p", "no:randomly",
                        "--no-cov", "--timeout", "300"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) failed", r.stdout); failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", r.stdout); passed = int(m.group(1)) if m else 0
    return failed, passed

MUJ = "strands_robots/simulation/mujoco/recording.py"
NEW_T = "strands_robots/simulation/newton/recording.py"
ISA = "strands_robots/simulation/isaac/recording.py"
GENERIC = 'unavailable = "the lerobot dataset stack is unavailable."'

MUTATIONS = [
    ("M1 mujoco: collapse the partial-install diagnosis into a generic one", MUJ, "start_recording",
     'unavailable = f"strands_robots.dataset_recorder is unavailable ({exc})."', GENERIC),
    ("M2 newton: collapse the partial-install diagnosis into a generic one", NEW_T, "start_recording",
     'unavailable = f"strands_robots.dataset_recorder is unavailable ({exc})."', GENERIC),
    ("M3 isaac: collapse the partial-install diagnosis into a generic one", ISA, "start_recording",
     'unavailable = f"strands_robots.dataset_recorder is unavailable ({exc})."', GENERIC),
    ("M4 isaac: drop the no-recorder-symbol check", ISA, "start_recording",
     'if unavailable is None and _DatasetRecorder is None:', 'if False and _DatasetRecorder is None:'),
    ("M5 newton: drop the no-recorder-symbol check", NEW_T, "start_recording",
     'if unavailable is None and _DatasetRecorder is None:', 'if False and _DatasetRecorder is None:'),
    ("M6 isaac: narrow the import guard to ModuleNotFoundError", ISA, "start_recording",
     "except ImportError as exc:", "except ModuleNotFoundError as exc:"),
    ("M7 newton: recommend a fallback it does not implement", NEW_T, "start_recording",
     '"For plain MP4 video, pass video={\'path\': ...} to run_policy instead."',
     '"For plain MP4 video, use start_cameras_recording instead."'),
]

print(f"\n{'mutation':62s} {'new tests':>12s} {'pre-existing':>14s}")
print("-" * 92)
rows = []
for label, path, fname, old, new in MUTATIONS:
    print(f"  {label}")
    orig = apply(path, fname, old, new)
    try:
        nf, np_ = run(NEW)
        of, op = run(OLD)
    finally:
        (ROOT/path).write_text(orig)
        assert (ROOT/path).read_text() == orig
    rows.append((label, nf, np_, of, op))
    print(f"{label[:60]:62s} {str(nf)+' failed':>12s} {str(of)+' failed':>14s}   (pre-existing passed={op})")

print("\n=== unmutated control ===")
nf, np_ = run(NEW); of, op = run(OLD)
print(f"{'control (no mutation)':62s} {str(nf)+' failed':>12s} {str(of)+' failed':>14s}   new passed={np_} old passed={op}")
caught_new = sum(1 for r in rows if r[1] > 0)
caught_old = sum(1 for r in rows if r[3] > 0)
print(f"\ncaught by the new module: {caught_new} of {len(rows)}")
print(f"caught by pre-existing:   {caught_old} of {len(rows)}")
assert nf == 0 and of == 0, "control must be green"
