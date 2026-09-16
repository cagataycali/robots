"""``scripts/ci_guards.py`` folds the pull-request guards into one step of the required check.

Ten advisory workflow rows became one step, so the properties that used to be
pinned per workflow are pinned here on the fold: every guard still runs and is
still named, the LLM-input scan fails on a finding instead of annotating one,
and a guard that cannot compute an answer is a broken check (exit 2), never a
finding (exit 1) and never a pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "ci_guards.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ci_guards", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_guards"] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_the_tree_carries_no_llm_input_finding() -> None:
    """The guard became a gate at zero findings; a new one is a real defect or a real false positive."""
    assert mod.scan_llm_input() == []


def test_a_subprocess_f_string_in_agent_callable_code_is_a_finding(tmp_path: Path) -> None:
    tools = tmp_path / "strands_robots" / "tools"
    tools.mkdir(parents=True)
    (tools / "bad.py").write_text('import subprocess\nsubprocess.run(f"ls {name}", shell=True)\n', encoding="utf-8")
    (tmp_path / "strands_robots" / "simulation").mkdir()
    (tmp_path / "strands_robots" / "mesh").mkdir()
    hits = mod.scan_llm_input(tmp_path)
    assert [(path, line) for path, line, _ in hits] == [("strands_robots/tools/bad.py", 2)]


def test_a_name_interpolated_into_mjcf_is_a_finding(tmp_path: Path) -> None:
    sim = tmp_path / "strands_robots" / "simulation"
    sim.mkdir(parents=True)
    (sim / "bad.py").write_text("xml = f\"<body pos='0 0 0' {body_name}/>\"\n", encoding="utf-8")
    (tmp_path / "strands_robots" / "tools").mkdir()
    (tmp_path / "strands_robots" / "mesh").mkdir()
    assert [(path, line) for path, line, _ in mod.scan_llm_input(tmp_path)] == [("strands_robots/simulation/bad.py", 1)]


def test_a_finding_fails_and_a_broken_guard_is_told_apart(capsys) -> None:
    clean = mod.Verdict("a", mod.CLEAN)
    finding = mod.Verdict("b", mod.FINDING)
    broken = mod.Verdict("c", mod.BROKEN, "exited 3")
    assert mod._report([clean], title="t") == 0
    assert mod._report([clean, finding], title="t") == 1
    assert mod._report([clean, finding, broken], title="t") == 2
    out = capsys.readouterr().out
    assert "::error title=Guards failed::b" in out
    assert "::error title=Guard broken::c" in out


def test_a_planted_finding_is_the_step_exit_status(monkeypatch) -> None:
    """A finding in one guard is what the step exits with, through the entry point.

    The cells above grade the scan, the verdict table and the status mapping
    separately, and all three still pass when ``guard_llm_input_safety`` returns
    its hits as a pass - which is precisely the behaviour being removed here
    (``llm-input-safety.yml`` annotated a match and ended in a step named
    "Always pass"). Only driving ``main`` closes that, so the revert is a failing
    test rather than a silent return to advisory.
    """
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    for name in ("guard_closing_reference", "guard_changelog_fragment", "guard_merge_base_overlap"):
        monkeypatch.setattr(mod, name, lambda *_, **__: mod.Verdict("stub", mod.CLEAN))
    monkeypatch.setattr(mod, "guard_lockfile_parity", lambda: mod.Verdict("stub", mod.CLEAN))
    argv = ["guards", "--event", "pull_request", "--head", "0" * 40, "--base-ref", "main"]

    monkeypatch.setattr(mod, "scan_llm_input", lambda *_: [])
    assert mod.main(argv) == mod.CLEAN

    monkeypatch.setattr(mod, "scan_llm_input", lambda *_: [("strands_robots/tools/bad.py", 2, "subprocess")])
    assert mod.main(argv) == mod.FINDING


def test_an_unexpected_exit_status_is_broken_not_a_finding() -> None:
    assert mod._as_verdict("x", 0).status == mod.CLEAN
    assert mod._as_verdict("x", 1).status == mod.FINDING
    assert mod._as_verdict("x", 2).status == mod.BROKEN
    assert mod._as_verdict("x", 127).status == mod.BROKEN


def test_the_guards_have_nothing_to_grade_outside_a_pull_request() -> None:
    assert mod.main(["guards", "--event", "push"]) == 0
    assert mod.main(["api-drift", "--event", "release"]) == 0


def test_a_pull_request_without_a_head_is_a_broken_check() -> None:
    assert mod.main(["guards", "--event", "pull_request", "--head", ""]) == mod.BROKEN


def test_every_folded_guard_is_still_named() -> None:
    """Deleting a guard from the fold must be a visible edit here, not a silent loss."""
    text = _SCRIPT.read_text(encoding="utf-8")
    for guard in (
        "check_closing_reference.py",
        "check_changelog_fragment.py",
        "check_merge_base_overlap.py",
        '"lock", "--check"',
        "scan_llm_input",
        "agent_api_snapshot.py",
        "griffe",
    ):
        assert guard in text, guard
    assert "Always pass" not in _REPO_ROOT.joinpath(".github", "workflows", "test-lint.yml").read_text(encoding="utf-8")
