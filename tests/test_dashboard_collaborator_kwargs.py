"""The Q56 audit, run as a test so it cannot rot in a scripts/ directory nobody executes.

Q56 was a call to `DatasetRecorder.push_to_hub(repo_id=...)` -- a parameter that class has never had.
It survived because every dashboard test injects a FAKE recorder: the only thing that could have
caught it was comparing the call site to the REAL signature, which is what this does.

Cheap (AST parse of dashboard/*.py + inspect on a handful of classes) and it fails with the exact
file, line, kwarg and real signature, so a break reads as an instruction.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "audit_collaborator_kwargs.py"


def _load():
    spec = importlib.util.spec_from_file_location("collab_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_the_dashboard_never_calls_a_collaborator_with_an_invented_kwarg():
    problems = _load().audit()
    assert not problems, "call sites the real classes would reject:\n" + "\n".join(problems)


def test_the_audit_actually_inspects_something():
    """A green audit that checked nothing is worse than no audit (it reassures)."""
    mod = _load()
    methods = mod.real_methods()
    assert "push_to_hub" in methods, "the Q56 method itself must be under audit"
    assert len(methods) > 20, f"only {len(methods)} methods resolved - collaborators failed to import"
    # The recorder receiver names must stay mapped, or the audit silently checks nothing there.
    assert mod.RECEIVERS["_recorder"][1] == "DatasetRecorder"


def test_it_matches_on_the_receiver_so_a_correct_call_is_not_flagged(tmp_path, monkeypatch):
    """`worker.close(upload=…, repo_id=…)` is right for RecordWorker and wrong for RecordController.

    Matching by method name alone flagged it - and three other innocents (PIL's save(format=),
    Path.open(encoding=), a module-level snapshot(bridge=)). A noisy audit gets muted, so this pins
    the receiver rule that made every hit trustworthy.
    """
    mod = _load()
    fake_dash = tmp_path / "dashboard"
    fake_dash.mkdir()
    (fake_dash / "ok.py").write_text(
        "def go(worker, img, path):\n"
        "    worker.close(upload=True, repo_id='x/y')\n"      # correct for RecordWorker
        "    img.save(format='JPEG', quality=80)\n"            # PIL, not ProfileStore
        "    return path.open(encoding='utf-8')\n"             # Path, not RecordController
    )
    monkeypatch.setattr(mod, "DASH", fake_dash)
    assert mod.audit() == []


def test_the_audit_can_fail(tmp_path, monkeypatch):
    """Plant a call site the real recorder would reject and prove the audit reports it."""
    mod = _load()
    fake_dash = tmp_path / "dashboard"
    fake_dash.mkdir()
    (fake_dash / "bogus.py").write_text(
        "def go(recorder):\n    return recorder.push_to_hub(repo_id='x/y')\n"
    )
    monkeypatch.setattr(mod, "DASH", fake_dash)
    problems = mod.audit()
    assert any("repo_id" in p and "push_to_hub" in p for p in problems), problems
