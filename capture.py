"""Measure every fact the figure states. Nothing in the figure is hand-typed."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "strands_robots/dataset_recorder.py"
NEW = "tests/test_lerobot_install_probe_contract.py"
OLD = [
    "tests/test_lerobot_dataset_import_diagnosis.py",
    "tests/test_lerobot_install_hints_pypi.py",
    "tests/test_dataset_recorder.py",
]
ENV = {**os.environ, "MUJOCO_GL": "egl"}
facts: dict[str, object] = {"tree": str(ROOT)}


def _pytest(paths: list[str], extra: list[str] | None = None) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "-p", "no:randomly", "--tb=no",
         *(extra or ["--no-cov"])],
        capture_output=True, text=True, cwd=ROOT, env=ENV,
    )
    return r.stdout


def _summary(out: str) -> str:
    for line in reversed(out.strip().splitlines()):
        if " passed" in line or " failed" in line:
            return line.strip("= ").strip()
    return "(none)"


def _recorder_cov(out: str) -> tuple[int, int, str]:
    for line in out.splitlines():
        if "strands_robots/dataset_recorder.py" in line:
            parts = line.split()
            return int(parts[2]), int(parts[3].rstrip("%")), line.split("%", 1)[1].strip()
    return (-1, -1, "")


# 1. the premise: does importing the recorder import lerobot?
proc = subprocess.run(
    [sys.executable, "-c", "import strands_robots.dataset_recorder, sys; print('lerobot' in sys.modules)"],
    capture_output=True, text=True, cwd=ROOT, env=ENV, check=True,
)
facts["fresh_process_lerobot_in_sys_modules"] = proc.stdout.strip()

# 2. coverage of the probe, before and after
cov_extra = ["--cov=strands_robots", "--cov-report=term-missing", "--cov-fail-under=0"]
before_out = _pytest(OLD, cov_extra)
after_out = _pytest([*OLD, NEW], cov_extra)
b_miss, b_pct, b_lines = _recorder_cov(before_out)
a_miss, a_pct, a_lines = _recorder_cov(after_out)
facts["cov_before"] = {"missing": b_miss, "percent": b_pct, "probe_lines_listed": "407-410" in b_lines}
facts["cov_after"] = {"missing": a_miss, "percent": a_pct, "probe_lines_listed": "407-410" in a_lines}
facts["subset_before"] = _summary(before_out)
facts["subset_after"] = _summary(after_out)

# 3. the mutation matrix
ORIG = TARGET.read_text()
lo, hi = next(
    (n.lineno, n.end_lineno)
    for n in ast.walk(ast.parse(ORIG))
    if isinstance(n, ast.FunctionDef) and n.name == "_lerobot_installed"
)
region = "".join(ORIG.splitlines(keepends=True)[lo - 1 : hi])
facts["probe_span"] = [lo, hi]

MUTS = [
    ("M1  the try/except is dropped",
     '    try:\n        return importlib.util.find_spec("lerobot") is not None\n    except (ImportError, ValueError):\n        return False\n',
     '    return importlib.util.find_spec("lerobot") is not None\n'),
    ("M2  an import replaces the spec lookup",
     '        return importlib.util.find_spec("lerobot") is not None\n',
     '        import lerobot  # noqa: F401\n\n        return True\n'),
    ("M3  the lookup outcome is ignored",
     '        return importlib.util.find_spec("lerobot") is not None\n',
     '        return True\n'),
    ("M4  the swallow drops ValueError",
     '    except (ImportError, ValueError):\n',
     '    except ImportError:\n'),
]
rows = []
try:
    for label, old, new in MUTS:
        assert region.count(old) == 1, label
        mutated = ORIG.replace(region, region.replace(old, new, 1), 1)
        ast.parse(mutated)
        TARGET.write_text(mutated)
        rows.append({"label": label, "new": _summary(_pytest([NEW])), "old": _summary(_pytest(OLD))})
finally:
    TARGET.write_text(ORIG)
    assert TARGET.read_text() == ORIG, "restore failed"
facts["mutations"] = rows
facts["restored_byte_identical"] = TARGET.read_text() == ORIG

out = pathlib.Path(f"/tmp/art-{pathlib.Path(ROOT).name}.json")
out.write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
print("WROTE", out)
