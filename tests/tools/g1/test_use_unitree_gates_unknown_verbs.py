"""``use_unitree`` gates a verb it has never heard of.

F-001 follow-up (CWE-862). The first fix put every *known* mutative verb behind
the operator gate, but the classifier deciding "does this need a prompt?" was a
denylist - ``_is_mutative`` answered ``True`` only for names starting with one
of 24 hardcoded prefixes. ``_execute`` dispatches any public attribute of the
SDK client and ``list_operations`` advertises every one, so a verb the pinned
SDK grows on its next bump (``Recover``, ``Trigger``, ``ArmTask``, ``Engage``)
was discoverable, dispatchable and ungated - and nothing tied the prefix table
to the SDK surface, so no test could turn red.

These tests grade the observable that matters - whether the stand-in client's
method ran - for names the prefix table has never seen. The classifier is now
an allowlist (read-only names pass; everything else asks), so an unknown verb
is the case the old code missed and the new code must catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.g1.use_unitree as uu

#: Plausible future SDK verbs, none starting with a MUTATIVE_PREFIXES entry.
UNKNOWN_VERBS = ("Frobnicate", "Recover", "Continue", "Trigger", "Activate", "Engage", "ArmTask", "DoThing")


class _Recorder:
    """Stand-in SDK client that records what ran instead of moving a robot."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def GetFsmId(self) -> tuple[int, int]:
        self.calls.append("GetFsmId")
        return (0, 801)

    def CheckMode(self) -> tuple[int, dict[str, str]]:
        self.calls.append("CheckMode")
        return (0, {"name": "ai"})


for _verb in UNKNOWN_VERBS:

    def _make(name: str):
        def _fn(self: _Recorder, **kwargs: Any) -> int:
            self.calls.append(name)
            return 0

        _fn.__name__ = name
        return _fn

    setattr(_Recorder, _verb, _make(_verb))


def _ctx(response: object) -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("BYPASS_TOOL_CONSENT", uu.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(uu, "ensure_dds", lambda _iface: None)
    monkeypatch.setattr(uu, "_CLIENTS", {"loco": rec, "arm": rec})
    return rec


class TestTheClassifierIsAnAllowlist:
    @pytest.mark.parametrize("verb", UNKNOWN_VERBS)
    def test_an_unknown_verb_is_mutative(self, verb: str) -> None:
        assert not any(verb.startswith(p) for p in uu.MUTATIVE_PREFIXES), "pick a verb the old denylist missed"
        assert uu._is_mutative(verb) is True
        assert uu._is_readonly(verb) is False

    @pytest.mark.parametrize("prefix", uu.MUTATIVE_PREFIXES)
    def test_every_documented_mutative_prefix_still_classifies_as_a_write(self, prefix: str) -> None:
        """The table is descriptive now, but it is also a floor the allowlist may not fall below."""
        assert uu._is_mutative(prefix) is True
        assert uu._is_mutative(prefix + "Something") is True

    @pytest.mark.parametrize("name", sorted(uu.READONLY_WHITELIST) + ["GetAnything", "CheckAnything"])
    def test_reads_stay_reads(self, name: str) -> None:
        assert uu._is_readonly(name) is True
        assert uu._is_mutative(name) is False

    def test_mutative_prefixes_are_not_consulted_by_the_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Emptying the table must not open the gate - the decision is made elsewhere."""
        monkeypatch.setattr(uu, "MUTATIVE_PREFIXES", ())
        assert uu._is_mutative("SetVelocity") is True
        assert uu._is_mutative("Frobnicate") is True
        assert uu._is_mutative("GetFsmId") is False


class TestAnUnknownVerbNeverReachesTheRobotUnapproved:
    @pytest.mark.parametrize("verb", UNKNOWN_VERBS)
    def test_declined_by_the_operator_it_is_not_dispatched(self, robot: _Recorder, verb: str) -> None:
        res = uu.use_unitree("loco", verb, {}, tool_context=_ctx("n"))

        assert res["status"] == "error", res
        assert res["dispatched"] is False
        assert res["mutative"] is True
        assert robot.calls == [], f"{verb} ran after the operator said no: {robot.calls}"

    @pytest.mark.parametrize("verb", UNKNOWN_VERBS)
    def test_headless_with_nothing_pre_approved_it_is_refused(self, robot: _Recorder, verb: str) -> None:
        res = uu.use_unitree("loco", verb, {})

        assert res["status"] == "error", res
        assert res["dispatched"] is False
        assert uu.COMMAND_ALLOW_ENV in res["message"]
        assert robot.calls == []

    def test_approved_by_the_operator_it_runs(self, robot: _Recorder) -> None:
        """The gate asks; it does not forbid. A ``y`` still lets a new verb through."""
        res = uu.use_unitree("loco", "Frobnicate", {}, tool_context=_ctx("y"))

        assert res["status"] == "success", res
        assert res["mutative"] is True
        assert robot.calls == ["Frobnicate"]

    def test_pre_approved_by_env_it_runs_without_a_prompt(
        self, robot: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(uu.COMMAND_ALLOW_ENV, "loco.Frobnicate")
        res = uu.use_unitree("loco", "Frobnicate", {})

        assert res["status"] == "success", res
        assert robot.calls == ["Frobnicate"]

    def test_a_read_is_still_never_gated(self, robot: _Recorder) -> None:
        """Headless, nothing pre-approved, no context: a read must still answer."""
        res = uu.use_unitree("loco", "GetFsmId", {})

        assert res["status"] == "success", res
        assert res["mutative"] is False
        assert robot.calls == ["GetFsmId"]

    def test_describe_operation_reports_an_unknown_verb_as_mutative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Discovery must tell the agent the truth the gate will act on."""
        monkeypatch.setattr(uu, "_import_client_class", lambda _q: _Recorder)
        desc = uu.describe_operation("loco", "Frobnicate")

        assert "error" not in desc, desc
        assert desc["is_mutative"] is True
        assert desc["is_readonly"] is False
