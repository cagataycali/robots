r"""On-disk text this package reads or writes is UTF-8, not the process locale.

``open()``, ``Path.read_text``/``write_text``, ``os.fdopen`` and
``tempfile.NamedTemporaryFile`` all decode and encode with
``locale.getencoding()`` when no ``encoding`` is passed, so a file whose bytes
are fixed answers differently in two processes that differ only in ``LC_ALL``.
On this package's files that is never the intent: a benchmark spec, a policy
config and a dataset's ``meta/info.json`` are UTF-8 documents - lerobot writes
``meta/info.json`` with ``encoding="utf-8"`` and ``ensure_ascii=False``, and a
spec is authored in an editor - so the reader that consults the locale is
reading a codec the writer never used.

Two things are graded here. :func:`test_no_on_disk_text_io_relies_on_the_locale`
scans the package so a new call site is graded on arrival; the subprocess cells
below pin what that buys, on three surfaces, under a locale that is not UTF-8.

The boundary is deliberate: this is about files on disk, not about a child
process's stdout (``subprocess(text=True)`` decodes with the locale because the
child encoded with it, which is a different decision).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots

PACKAGE = Path(strands_robots.__file__).parent

# A mode with "b" is binary: passing encoding= there is a TypeError, so those
# calls are not part of this rule.
TEXT_MODES = frozenset({"", "r", "w", "a", "x", "r+", "w+", "a+", "rt", "wt", "at"})
# Receiver names that make a bare ``.open()`` a file open rather than one of the
# many unrelated ``open`` verbs the package calls (zenoh sessions, PIL images,
# av containers, this package's own ``Reader.open`` classmethods).
PATHISH = ("path", "file", "_p")


def _mode_of(node: ast.Call, positional_index: int) -> str | None:
    """The literal mode a call states, or None when it states none."""
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value if isinstance(keyword.value.value, str) else None
    if len(node.args) > positional_index:
        arg = node.args[positional_index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _locale_dependent(node: ast.Call) -> str | None:
    """The construct name when this call decides an encoding by locale, else None.

    Returns None for a call that states ``encoding=``, for binary modes, and for
    the ``open`` verbs that are not file opens.
    """
    func = node.func
    if any(keyword.arg == "encoding" for keyword in node.keywords):
        return None
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)

    if name in ("read_text", "write_text") and isinstance(func, ast.Attribute):
        return name
    if name == "open" and isinstance(func, ast.Name):  # builtins.open(path, mode)
        return "open" if (_mode_of(node, 1) or "") in TEXT_MODES else None
    if name == "fdopen":  # os.fdopen(fd, mode) - mode defaults to "r"
        return "os.fdopen" if (_mode_of(node, 1) or "") in TEXT_MODES else None
    if name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"):
        mode = _mode_of(node, 0)  # default "w+b" is binary
        return f"tempfile.{name}" if mode is not None and mode in TEXT_MODES else None
    if name == "open" and isinstance(func, ast.Attribute):  # Path.open([mode])
        mode = _mode_of(node, 0)
        if mode is not None:
            return "Path.open" if mode in TEXT_MODES else None
        receiver = ast.unparse(func.value).lower()
        return "Path.open" if not node.args and any(p in receiver for p in PATHISH) else None
    return None


def _scan() -> tuple[list[str], int]:
    """(sites that decide by locale, count of sites that state an encoding)."""
    unstated: list[str] = []
    stated = 0
    for module in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if any(keyword.arg == "encoding" for keyword in node.keywords):
                stated += 1
                continue
            if construct := _locale_dependent(node):
                rel = module.relative_to(PACKAGE.parent)
                unstated.append(f"{rel}:{node.lineno} [{construct}] {ast.unparse(node)[:90]}")
    return unstated, stated


def test_no_on_disk_text_io_relies_on_the_locale():
    """Every text read/write in the package states its encoding."""
    unstated, stated = _scan()
    # Population check: without it a matcher that stops recognising these calls
    # would pass this test by finding nothing at all.
    assert stated > 40, f"scan recognised only {stated} calls that state an encoding; the matcher is broken"
    assert unstated == [], "these calls decode/encode with locale.getencoding():\n" + "\n".join(unstated)


# --------------------------------------------------------------------------- #
# What the rule buys, measured. The child is pure ASCII (\uXXXX escapes) so its
# source and argv survive an ASCII locale; the values it writes are not, and it
# writes them as UTF-8 the way the tools that produce these files do.
# --------------------------------------------------------------------------- #
INSTRUCTION = "coloca el cubo azul en la caja peque\u00f1a"
POLICY_PATH = "/models/moc\u00e7\u00e3o/gear.onnx"
TASKS = ["\u30de\u30b0\u3092\u62bc\u3059", "place the cup", "wipe the tray"]
TOTAL_TASKS = 3

CHILD = r"""
import json, locale, sys
from pathlib import Path

work = Path(sys.argv[1])
INSTRUCTION = "coloca el cubo azul en la caja peque\u00f1a"
POLICY_PATH = "/models/moc\u00e7\u00e3o/gear.onnx"
TASKS = ["\u30de\u30b0\u3092\u62bc\u3059", "place the cup", "wipe the tray"]

out = {"locale_encoding": locale.getencoding(), "surfaces": {}}


def record(name, fn):
    try:
        out["surfaces"][name] = {"ok": True, "value": fn()}
    except BaseException as exc:
        out["surfaces"][name] = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:160]}


def benchmark_instruction():
    from strands_robots.simulation.benchmark_spec import register_benchmark_from_file

    spec = work / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "default_robot": "so100",
                "supported_robots": ["so100"],
                "instruction": INSTRUCTION,
                "max_steps": 10,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return register_benchmark_from_file("locale_probe", spec).instruction


def wbc_policy_path():
    from strands_robots.policies.wbc.config import WBCConfig

    cfg = work / "wbc.json"
    cfg.write_text(json.dumps({"policy_path": POLICY_PATH}, ensure_ascii=False), encoding="utf-8")
    return WBCConfig.from_file(cfg).policy_path


def dataset_total_tasks():
    from strands_robots.training.lerobot import LerobotTrainer

    meta = work / "ds" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    # Exactly how lerobot.utils.io_utils.write_json writes meta/info.json.
    with open(meta / "info.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"codebase_version": "v3.0", "total_episodes": 9, "total_tasks": 3, "tasks": TASKS},
            handle,
            indent=4,
            ensure_ascii=False,
        )
    return LerobotTrainer()._dataset_total_tasks(str(work / "ds"))


record("benchmark_instruction", benchmark_instruction)
record("wbc_policy_path", wbc_policy_path)
record("dataset_total_tasks", dataset_total_tasks)
print(json.dumps(out))  # ASCII stdout: this process may have an ASCII locale
"""


def _probe(tmp_path: Path, locale_name: str) -> dict[str, Any]:
    """Run the child under ``locale_name`` with UTF-8 mode off, return its verdicts."""
    work = tmp_path / locale_name.replace(".", "_")
    work.mkdir(parents=True, exist_ok=True)
    script = work / "probe.py"
    script.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ, LC_ALL=locale_name, LANG=locale_name)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", str(script), str(work)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def under_ascii_locale(tmp_path_factory) -> dict[str, Any]:
    """Verdicts from a process whose locale encoding is not UTF-8."""
    report = _probe(tmp_path_factory.mktemp("ascii"), "C")
    if report["locale_encoding"].lower().replace("-", "") in ("utf8", "ansix341968utf8"):
        pytest.skip(f"no non-UTF-8 locale available here (got {report['locale_encoding']})")
    return report


@pytest.fixture(scope="module")
def under_utf8_locale(tmp_path_factory) -> dict[str, Any]:
    """Verdicts from a UTF-8 process: the control this change must not move."""
    return _probe(tmp_path_factory.mktemp("utf8"), "C.utf8")


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        # A spec's instruction is the language command handed to the policy.
        ("benchmark_instruction", INSTRUCTION),
        # A config's policy_path names a directory that may not be ASCII.
        ("wbc_policy_path", POLICY_PATH),
        # A header lerobot wrote as UTF-8. Read as absent, validation_split_error
        # treats the dataset as single-task and admits a split it must refuse.
        ("dataset_total_tasks", TOTAL_TASKS),
    ],
)
def test_a_utf8_file_reads_the_same_under_a_non_utf8_locale(under_ascii_locale, surface, expected):
    result = under_ascii_locale["surfaces"][surface]
    assert result["ok"], f"{surface} escaped under locale {under_ascii_locale['locale_encoding']}: {result}"
    assert result["value"] == expected


def test_a_utf8_locale_reads_exactly_the_same_values(under_utf8_locale, under_ascii_locale):
    """The locale stops deciding; it does not start deciding differently."""
    assert under_utf8_locale["surfaces"] == under_ascii_locale["surfaces"]
