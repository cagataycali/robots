"""A caller-supplied peer_id is a zenoh KEY SEGMENT, not a label (Q3 corollary).

``/api/devices/spawn`` accepted ``peer_id`` verbatim into every topic built from
it. In a zenoh key expression ``*`` and ``**`` are WILDCARDS and ``/`` is the
hierarchy separator - a peer named ``*`` shadows the whole fleet's key space.
These tests pin the refusal at the validator and at ``spawn()`` itself, BEFORE
any process exists (the dangerous ids must never reach Popen, let alone zenoh).
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.device_manager import (
    DeviceManager,
    validate_peer_id,
)


class TestValidatePeerId:
    def test_none_means_generate_one_and_is_fine(self) -> None:
        assert validate_peer_id(None) is None

    @pytest.mark.parametrize(
        "good",
        [
            "so101-arm-1",
            "replay-1234",
            "so101.real:2",
            "A_z".replace("_", "-"),
            "x" * 64,
        ],
    )
    def test_every_id_this_codebase_generates_passes(self, good: str) -> None:
        assert validate_peer_id(good) is None, good

    @pytest.mark.parametrize(
        "bad",
        [
            "*",  # fleet-wide wildcard
            "**",  # multi-level wildcard
            "so101/*",  # wildcard inside a hierarchy
            "a/b",  # splices a key level
            "../../../tmp/pwn",
            "so101 arm",  # space
            "{a:1}",  # brace DSL
            "$now",  # dollar DSL
            "",  # empty
            "x" * 65,  # over-long
        ],
    )
    def test_key_expression_syntax_is_refused_with_the_reason(self, bad: str) -> None:
        reason = validate_peer_id(bad)
        assert reason is not None, f"{bad!r} was accepted into a zenoh key"
        assert "zenoh" in reason or "string" in reason

    @pytest.mark.parametrize("nonstring", [123, {"a": 1}, ["x"], 1.5, True])
    def test_a_non_string_is_refused_by_type_not_repred(self, nonstring: object) -> None:
        reason = validate_peer_id(nonstring)
        assert reason is not None
        assert "must be a string" in reason


class TestSpawnRefusesBeforeAnyProcessExists:
    def test_a_wildcard_peer_id_never_reaches_popen(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("Popen was reached with a wildcard peer_id")

        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(mod.subprocess, "Popen", boom)
        result = dm.spawn("so101", "sim", peer_id="*")
        assert "error" in result
        assert "zenoh" in result["error"]
        assert dm.robots == {}, "a refused spawn must not register a managed entry"

    def test_a_generated_id_still_spawns_normally(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # peer_id=None takes the generated-id path: validation must not get in
        # its way. Popen is stubbed so no child is created.
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))

        class FakeProc:
            pid = 4242
            stdout = None

            def poll(self):
                return None

        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: FakeProc())
        monkeypatch.setattr(mod.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
        result = dm.spawn("so101", "sim", peer_id=None, remember=False)
        assert result.get("pid") == 4242, result
        assert validate_peer_id(result["peer_id"]) is None, (
            "generated ids must satisfy the same rule callers are held to"
        )
