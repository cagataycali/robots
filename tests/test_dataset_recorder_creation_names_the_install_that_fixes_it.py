"""Creating a recorder reports the same import diagnosis the probe reports.

``lerobot.datasets.lerobot_dataset`` fails to import for four unrelated reasons
and they need four different instructions, so
:mod:`strands_robots.dataset_recorder` composes one diagnosis
(:func:`~strands_robots.dataset_recorder.lerobot_dataset_import_error`) naming
which cause applied and the install that fixes it - or that no install does.
Every backend's ``start_recording`` reports that diagnosis, and
``tests/test_lerobot_install_hints_pypi.py`` pins its contract: two causes are
answered with a lerobot install and the other two deliberately are not, because
lerobot is already installed for both.

The documented direct creation API - ``DatasetRecorder.create`` and ``.resume`` -
imported the same module for the same reasons and composed a second,
contradicting answer: "lerobot not available. Install with: pip install
lerobot". For an absent dataset dependency that names a package that is already
there, so following it changes nothing; for a conflict between installed
packages no install is the fix at all. These cells hold both surfaces to one
diagnosis, so the contract pinned on the probe covers the creation API too.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from strands_robots import dataset_recorder as dr

#: A remedy naming the bare ``lerobot`` distribution, which supplies the package
#: but not its dataset stack. The lookahead lets ``pip install 'lerobot[dataset]'``
#: and ``pip install 'strands-robots[lerobot]'`` through.
_BARE_LEROBOT_INSTALL = re.compile(r"pip install (?:-U )?lerobot(?![\w\[.-])")


class _Case(NamedTuple):
    """One failed import and the answer it must get.

    Attributes:
        exc: The exception the dataset-stack import raises.
        installed: Whether lerobot itself is present, the premise the diagnosis
            branches on (patched, so the case under test is the one named here
            on every install).
        version: The lerobot version to report.
        remedy: The install that fixes this cause, or None when no install does.
        names: A detail the message must carry, so the caller can act on it.
    """

    exc: BaseException
    installed: bool
    version: str
    remedy: str | None
    names: str


_CASES = {
    "lerobot_absent": _Case(
        ModuleNotFoundError("no lerobot", name="lerobot"),
        False,
        "",
        "pip install 'strands-robots[lerobot]'",
        "not installed",
    ),
    "dataset_dependency_absent": _Case(
        ModuleNotFoundError("no pyarrow", name="pyarrow"),
        True,
        "0.6.1",
        "pip install 'lerobot[dataset]'",
        "pyarrow",
    ),
    "module_moved": _Case(
        ImportError("gone", name="lerobot.datasets.lerobot_dataset"),
        True,
        "0.6.1",
        "pip install 'strands-robots[lerobot]'",
        "lerobot.datasets.lerobot_dataset",
    ),
    "packages_conflict": _Case(
        ValueError("numpy.dtype size changed"),
        True,
        "0.6.1",
        None,
        "reconcile the conflicting packages",
    ),
}


def _assert_answers(message: str, case: _Case) -> None:
    """Check *message* against the one answer *case* may be given."""
    assert case.names in message, message
    assert _BARE_LEROBOT_INSTALL.search(message) is None, message
    if case.remedy is None:
        # Nothing is absent, so no install is offered rather than one that cannot help.
        assert "pip install" not in message, message
    else:
        assert case.remedy in message, message


class _UnusableLerobot:
    """A ``lerobot`` whose every import raises, recording that it was reached."""

    def __init__(self, case: _Case) -> None:
        self.case = case
        self.attempts = 0

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "lerobot" or name.startswith("lerobot."):
            self.attempts += 1
            raise self.case.exc
        return None


@pytest.fixture
def broken(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _UnusableLerobot:
    """Make the dataset-stack import fail with the named case's exception."""
    case = _CASES[request.param]
    for name in [m for m in list(sys.modules) if m == "lerobot" or m.startswith("lerobot.")]:
        monkeypatch.delitem(sys.modules, name)
    breaker = _UnusableLerobot(case)
    monkeypatch.setattr(sys, "meta_path", [breaker, *sys.meta_path])
    monkeypatch.setattr(dr, "_lerobot_installed", lambda: case.installed)
    monkeypatch.setattr(dr, "lerobot_version", lambda: case.version)
    # A successful probe is cached; the cache has to be empty for the probe to
    # attempt the import this fixture breaks.
    monkeypatch.setattr(dr, "_HAS_LEROBOT_DATASET", [])
    monkeypatch.setattr(dr, "LeRobotDataset", None, raising=False)
    return breaker


def _create(root: Path) -> str:
    """Return the message ``DatasetRecorder.create`` refuses with."""
    with pytest.raises(ImportError) as excinfo:
        dr.DatasetRecorder.create(
            repo_id="probe/ds",
            fps=30,
            robot_type="so100",
            camera_keys=[],
            joint_names=["shoulder_pan"],
            task="pick",
            root=str(root),
        )
    return str(excinfo.value)


@pytest.mark.parametrize("broken", list(_CASES), indirect=True)
def test_the_probe_answers_each_cause_with_its_own_remedy(broken: _UnusableLerobot) -> None:
    """The reference contract the cells below lean on: four causes, four answers.

    Asserted rather than assumed - a probe that gave one answer to every cause
    would make "the two surfaces agree" cost nothing.
    """
    diagnosis = dr.lerobot_dataset_import_error()
    assert diagnosis, "premise: the probe did not report the broken import"
    _assert_answers(diagnosis, broken.case)


@pytest.mark.parametrize("broken", list(_CASES), indirect=True)
def test_create_reports_what_the_probe_reports(broken: _UnusableLerobot, tmp_path: Path) -> None:
    """One diagnosis, whichever surface the caller reached it through."""
    diagnosis = dr.lerobot_dataset_import_error()
    assert diagnosis, "premise: the probe did not report the broken import"

    message = _create(tmp_path / "ds")
    assert message.startswith(diagnosis), (
        "DatasetRecorder.create composed its own answer instead of the one the "
        f"probe reports.\ncreate: {message!r}\nprobe:  {diagnosis!r}"
    )
    _assert_answers(message, broken.case)
    assert broken.attempts, "premise: the dataset stack import was never reached"


@pytest.mark.parametrize("broken", ["dataset_dependency_absent"], indirect=True)
def test_resume_reports_what_the_probe_reports(broken: _UnusableLerobot, tmp_path: Path) -> None:
    """The append entry point reads the same diagnosis as ``create``."""
    diagnosis = dr.lerobot_dataset_import_error()
    assert diagnosis, "premise: the probe did not report the broken import"

    with pytest.raises(ImportError) as excinfo:
        dr.DatasetRecorder.resume(repo_id="probe/ds", root=str(tmp_path / "ds"), task="pick")
    assert str(excinfo.value).startswith(diagnosis), str(excinfo.value)
    _assert_answers(str(excinfo.value), broken.case)


def test_the_recorder_writes_one_lerobot_install_remedy() -> None:
    """No string in the module offers the install its own diagnosis rules out.

    The duplicate was two functions importing the same module, so the pin is on
    the module's strings rather than on the one call site that had it: the
    docstring explaining why a bare lerobot install is not the instruction is the
    only place that spelling belongs.
    """
    tree = ast.parse(Path(dr.__file__).read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in [tree, *ast.walk(tree)]
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    offenders: list[str] = []
    explained = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        text = re.sub(r"\s+", " ", node.value)
        if not _BARE_LEROBOT_INSTALL.search(text):
            continue
        if id(node) in docstrings:
            explained += 1
            continue
        offenders.append(f"line {node.lineno}: {text.strip()[:120]}")
    assert explained, "the matcher found no mention at all, so it is not reading the module"
    assert not offenders, (
        "these strings tell a caller to install lerobot, which the module's own "
        "diagnosis records as not the fix for three of its four causes:\n" + "\n".join(offenders)
    )
