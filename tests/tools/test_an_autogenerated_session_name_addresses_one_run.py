"""An auto-generated session name addresses the run it was handed back for.

``start`` hands the caller a session name and that name is their only handle on
the detached process: ``status`` reports it, ``stop`` signals it. A name derived
from a second-resolution clock alone is not that handle, because two ``start``
calls in the same second derive the same one - the ordinary case for an agent
issuing independent tool calls together. The second record then replaces the
first under the shared name, so the name returned for one run addresses the
other, the first run becomes unreachable by every verb, and both runs append to
the one log file the name resolves to.

:mod:`strands_robots.tools._process_stop` already refuses that outcome when it
arrives through a reused pid (its module docstring: a session record pointing at
"whatever holds the number next" reports a stranger and invites ``stop`` to
signal it). These pin the same refusal when it arrives through a reused name,
for both session tools, and pin that a name the caller chose themselves is still
theirs: an explicit collision is reported, not silently renamed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("psutil")

import strands_robots.tools.lerobot_teleoperate as teleop_mod  # noqa: E402
import strands_robots.tools.lerobot_train as train_mod  # noqa: E402
from strands_robots.tools import _process_stop  # noqa: E402
from tests.tool_result_contract import tool_json  # noqa: E402

FROZEN = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Start from an empty store in a temp dir, and freeze the clock.

    The clock is frozen because a second-resolution name is only ambiguous
    within one tick; freezing it makes the collision the deterministic case
    instead of one that depends on how fast two calls land.
    """
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)
    monkeypatch.setattr(_process_stop.time, "time", lambda: FROZEN)
    return session_dir


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def _dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return root


def _start_two_trainings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Two auto-named training starts under one clock tick, distinct pids."""
    root = _dataset(tmp_path / "ds")
    pids = iter((4242, 4243))
    monkeypatch.setattr(train_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(next(pids)))
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)
    return [
        train_mod.lerobot_train(
            action="start",
            dataset_root=str(root),
            policy_type=policy,
            output_dir=str(tmp_path / f"out_{policy}"),
        )
        for policy in ("act", "diffusion")
    ]


class TestTwoStartsInOneTickStayReachable:
    """Neither run loses its handle to the other."""

    def test_both_starts_succeed_with_distinct_names(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first, second = _start_two_trainings(tmp_path, monkeypatch)
        assert first["status"] == "success", first
        assert second["status"] == "success", second
        a, b = tool_json(first)["session_name"], tool_json(second)["session_name"]
        assert a.startswith("train_") and b.startswith("train_")
        assert a != b, f"both runs were handed the same handle: {a}"

    def test_each_name_addresses_its_own_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """status under each returned name reports that run's own pid and policy."""
        results = _start_two_trainings(tmp_path, monkeypatch)
        seen: dict[str, tuple[object, object]] = {}
        for result, policy, pid in zip(results, ("act", "diffusion"), (4242, 4243), strict=True):
            name = tool_json(result)["session_name"]
            record = train_mod.SessionManager().get_session(name)
            assert record is not None, f"{name} reaches no record"
            seen[name] = (record["pid"], record["policy_type"])
            assert record["pid"] == pid
            assert record["policy_type"] == policy
        assert len(seen) == 2, f"the two runs collapsed onto one record: {seen}"

    def test_the_two_runs_do_not_share_one_log_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shared name resolves to a shared log, interleaving both runs."""
        names = [tool_json(r)["session_name"] for r in _start_two_trainings(tmp_path, monkeypatch)]
        logs = {_process_stop.session_log_path(n) for n in names}
        assert len(logs) == 2, f"both runs captured output into {logs}"

    def test_list_reports_both_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _start_two_trainings(tmp_path, monkeypatch)
        listed = train_mod.lerobot_train(action="list", dataset_root=str(tmp_path / "ds"))
        assert tool_json(listed)["count"] == 2


class TestAnExplicitNameStaysTheCallersOwn:
    """A caller who names a session owns the name; it is neither renamed nor stolen."""

    def test_an_explicit_name_is_used_verbatim(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = _dataset(tmp_path / "ds")
        monkeypatch.setattr(train_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(4242))
        monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)
        result = train_mod.lerobot_train(
            action="start",
            dataset_root=str(root),
            policy_type="act",
            output_dir=str(tmp_path / "out"),
            session_name="my_run",
        )
        assert tool_json(result)["session_name"] == "my_run"

    def test_an_explicit_collision_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = _dataset(tmp_path / "ds")
        monkeypatch.setattr(train_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(4242))
        monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)
        train_mod.lerobot_train(
            action="start",
            dataset_root=str(root),
            policy_type="act",
            output_dir=str(tmp_path / "out"),
            session_name="taken",
        )
        again = train_mod.lerobot_train(
            action="start",
            dataset_root=str(root),
            policy_type="act",
            output_dir=str(tmp_path / "out2"),
            session_name="taken",
        )
        assert again["status"] == "error"
        assert "already exists" in "".join(item.get("text", "") for item in again["content"] if "text" in item)


@pytest.mark.parametrize(
    ("module", "prefix"),
    [(train_mod, "train"), (teleop_mod, "teleop")],
    ids=["train", "teleop"],
)
def test_both_session_tools_name_a_run_through_the_one_generator(module: ModuleType, prefix: str) -> None:
    """One owner decides the shape, so neither tool can drift back to the clock."""
    assert module.generate_session_name is _process_stop.generate_session_name
    name = _process_stop.generate_session_name(prefix)
    assert name.startswith(f"{prefix}_{int(FROZEN)}_")
    assert name != _process_stop.generate_session_name(prefix)
