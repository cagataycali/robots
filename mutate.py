"""Mutation table for GH #2239: 7 plausible regressions x 2 test arms."""
from __future__ import annotations
import pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NEW = "tests/test_abandoned_work_item_cannot_hang_exit.py"
PRE = [
    "tests/test_hardware_policy_port_domain.py",
    "tests/test_hardware_robot_lifecycle.py",
    "tests/test_hardware_cleanup_disconnects.py",
    "tests/test_hardware_action_horizon_guard.py",
    "tests/test_hardware_policy_per_task_reset.py",
]
HELPER = "tests/_daemon_executor.py"

POOL = 'DaemonThreadExecutor(max_workers=1, thread_name_prefix="test_arm_executor")'
POOL_OLD = 'ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")'
IMP_NEW = "from tests._daemon_executor import DaemonThreadExecutor"
IMP_OLD = "from concurrent.futures import ThreadPoolExecutor"

MUTATIONS = [
    ("M1 port_domain fixture back to ThreadPoolExecutor",
     "tests/test_hardware_policy_port_domain.py",
     [(IMP_NEW, IMP_OLD), (POOL, POOL_OLD)]),
    ("M2 lifecycle fixture back to ThreadPoolExecutor",
     "tests/test_hardware_robot_lifecycle.py",
     [(IMP_NEW, IMP_OLD), (POOL, POOL_OLD)]),
    ("M3 cleanup_disconnects fixture back to ThreadPoolExecutor",
     "tests/test_hardware_cleanup_disconnects.py",
     [(IMP_NEW, IMP_OLD), (POOL, POOL_OLD)]),
    ("M4 helper worker is not a daemon", HELPER, [("daemon=True", "daemon=False")]),
    ("M5 shutdown(wait=True) does not join", HELPER,
     [("        if wait and worker is not None:\n            worker.join()",
       "        if False and worker is not None:\n            worker.join()")]),
    ("M6 submit after shutdown is allowed", HELPER,
     [('                raise RuntimeError("cannot schedule new futures after shutdown")',
       "                pass")]),
    ("M7 worker swallows the exception", HELPER,
     [("                future.set_exception(exc)", "                future.set_result(None)")]),
]


def run(paths: list[str], budget: int) -> str:
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
            cwd=ROOT, capture_output=True, text=True, timeout=budget, check=False,
        )
    except subprocess.TimeoutExpired:
        return "TIMED OUT"
    tail = [ln for ln in done.stdout.splitlines() if re.match(r"^=+.*(passed|failed|error)", ln)]
    if not tail:
        return "no summary"
    line = tail[-1]
    f = re.search(r"(\d+) failed", line)
    e = re.search(r"(\d+) error", line)
    p = re.search(r"(\d+) passed", line)
    return f"{int(f.group(1)) if f else 0} failed, {int(e.group(1)) if e else 0} error, {int(p.group(1)) if p else 0} passed"


print(f"{'mutation':<48} {'new module':<34} {'pre-existing (201)':<34}")
print("-" * 118)
print(f"{'(unmutated control)':<48} {run([NEW], 400):<34} {run(PRE, 400):<34}")

tmp = pathlib.Path(tempfile.mkdtemp())
for label, path, edits in MUTATIONS:
    p = ROOT / path
    saved = tmp / p.name
    shutil.copy2(p, saved)
    try:
        src = p.read_text()
        for old, new in edits:
            n_file = src.count(old)
            assert n_file == 1, f"{label}: anchor appears {n_file}x in {path}: {old!r}"
            src = src.replace(old, new)
        p.write_text(src)
        print(f"{label:<48} {run([NEW], 400):<34} {run(PRE, 400):<34}")
    finally:
        shutil.copy2(saved, p)
        assert p.read_text() == saved.read_text()
print("-" * 118)
print("restored byte-identically:", all((ROOT / m[1]).read_text() == (tmp / (ROOT / m[1]).name).read_text() for m in MUTATIONS))
