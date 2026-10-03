# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the ``run_policy`` tool judges the policy before it touches disk.

The tool owns the recording lifecycle and starts it with ``overwrite=True``, so
anything the rollout refuses inside the episode loop is refused AFTER the
dataset already at ``dataset_root`` has been replaced with an empty one. The
tool's own pre-flight block guards every knob it forwards - rates, seed, video,
the keyword bags, ``stop_when`` - except the one that names what will actually
run: the policy. Measured before the fix against a dataset of two episodes and
ten frames, an unknown provider and a ``lerobot_local`` checkpoint id that does
not exist each returned ``run_policy: 0/1 episodes ok | parquet-truth:
total_episodes=0`` and left ``meta/info.json`` reading ``0 0``.

Now, when a recording is requested, the tool resolves the provider, runs the
shared :func:`~strands_robots.policies.preflight_reason` against the
simulation's observation keys and builds the policy ONCE, all before
``start_recording``; the built policy is handed to every episode as
``policy_object`` (so a two-minute checkpoint load is paid once per call, not
once per episode). The recording-less path forwards exactly what it forwarded
before. A rollout that still fails inside the loop names its first reason on
the summary line, where an agent reads it, instead of only inside the
per-episode records.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

import strands_robots.policies as policies_pkg
from strands_robots.policies import list_providers
from strands_robots.policies.mock import MockPolicy
from tests.tools.test_run_policy import _FakeSim, _ok_rollout

rp_mod = importlib.import_module("strands_robots.tools.run_policy")

ROOT = "/tmp/run-policy-policy-preflight"


def _run_tool(sim: Any, **kwargs: Any) -> dict[str, Any]:
    return dict(rp_mod.run_policy(sim, **kwargs))


def _text(result: dict[str, Any]) -> str:
    return str((result.get("content") or [{}])[0].get("text", ""))


class _OrderedSim(_FakeSim):
    """A fake that also keeps the ORDER of the calls the tool makes."""

    def __init__(self) -> None:
        super().__init__()
        self.order: list[str] = []
        self.observation_reads = 0

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        self.observation_reads += 1
        return {"1": 0.0, "2": 0.0, "front": object(), "wrist": object()}

    def run_policy(self, **kwargs: Any) -> dict[str, Any]:
        self.order.append("run_policy")
        return super().run_policy(**kwargs)

    def start_recording(self, **kwargs: Any) -> dict[str, Any]:
        self.order.append("start_recording")
        return super().start_recording(**kwargs)

    def stop_recording(self, **kwargs: Any) -> dict[str, Any]:
        self.order.append("stop_recording")
        return super().stop_recording(**kwargs)


class TestAnUnresolvableProviderIsRefusedBeforeTheRecording:
    """The provider name is judged by the same rule ``create_policy`` applies."""

    def test_an_unknown_provider_starts_no_recording(self) -> None:
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="no_such_provider_xyz", n_steps=4, dataset_root=ROOT)
        assert result["status"] == "error"
        assert "no_such_provider_xyz" in _text(result)
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []
        assert sim.stop_recording_calls == []

    def test_the_refusal_names_the_shipped_providers(self) -> None:
        """The words are ``policy_provider_error``'s: the shipped registry, offered by name.

        Pinned to names the JSON registry ships rather than to ``list_providers()``,
        which also reports providers a sibling test registered at runtime for
        the length of the session and the refusal does not offer.
        """
        result = _run_tool(_OrderedSim(), policy_provider="no_such_provider_xyz", n_steps=4, dataset_root=ROOT)
        assert "Available:" in _text(result)
        for name in ("mock", "lerobot_local", "remote", "wbc"):
            assert name in list_providers()
            assert f"'{name}'" in _text(result), f"the refusal does not offer {name!r}"

    def test_the_summary_says_the_dataset_is_untouched(self) -> None:
        result = _run_tool(_OrderedSim(), policy_provider="no_such_provider_xyz", n_steps=4, dataset_root=ROOT)
        assert ROOT in _text(result)
        assert "untouched" in _text(result)


class TestAConstructionFailureIsRefusedBeforeTheRecording:
    """What only the constructor can judge is judged before the wipe, as an envelope.

    ``lerobot_local`` with a checkpoint id that does not exist is the measured
    case: its constructor raises ``FileNotFoundError`` out of the Hub lookup.
    The Hub is not dialled here; the factory is replaced by one that raises
    what the real one raised, so the test grades the tool's handling of the
    raise rather than the network.
    """

    @pytest.mark.parametrize(
        "raised",
        [
            FileNotFoundError("config.json not found on the HuggingFace Hub"),
            ValueError("pretrained_name_or_path: not a checkpoint"),
            RuntimeError("WBCPolicy main ONNX checkpoint not found"),
            ConnectionError("RemotePolicy could not reach a PolicyServer at ws://127.0.0.1:59999"),
        ],
        ids=type,
    )
    def test_a_constructor_raise_is_an_envelope_and_starts_no_recording(
        self, monkeypatch: pytest.MonkeyPatch, raised: Exception
    ) -> None:
        def refusing_factory(provider: str, **kwargs: Any) -> Any:
            raise raised

        monkeypatch.setattr(policies_pkg, "create_policy", refusing_factory)
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="mock", n_episodes=2, n_steps=4, dataset_root=ROOT)
        assert result["status"] == "error"
        assert str(raised) in _text(result)
        assert "'mock'" in _text(result)
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []
        assert sim.stop_recording_calls == []


class TestTheProvidersPreflightRunsBeforeTheRecording:
    """The shared pre-build hook reads THIS simulation's observation keys."""

    def test_a_preflight_refusal_starts_no_recording(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def refusing_preflight(provider: str, read_keys: Any, /, **kwargs: Any) -> str | None:
            seen["provider"] = provider
            seen["keys"] = set(read_keys())
            seen["kwargs"] = kwargs
            return "camera 'top' is not among the observation keys"

        monkeypatch.setattr(policies_pkg, "preflight_reason", refusing_preflight)
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="mock", policy_config={"camera": "top"}, n_steps=4, dataset_root=ROOT)
        assert result["status"] == "error"
        assert "camera 'top' is not among the observation keys" in _text(result)
        assert seen == {"provider": "mock", "keys": {"1", "2", "front", "wrist"}, "kwargs": {"camera": "top"}}
        assert sim.start_recording_calls == []
        assert sim.run_policy_calls == []

    def test_a_provider_without_a_hook_reads_no_observation(self) -> None:
        """``mock`` leaves the default no-op in place, so no camera is rendered for it."""
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="mock", n_steps=4, dataset_root=ROOT)
        assert result["status"] in {"success", "error"}
        assert sim.observation_reads == 0

    def test_a_simulation_without_get_observation_still_runs(self) -> None:
        """The reader is optional: a stand-in with no observation surface is not refused."""
        sim = _FakeSim()
        result = _run_tool(sim, policy_provider="mock", n_steps=4, dataset_root=ROOT)
        assert len(sim.start_recording_calls) == 1
        assert len(sim.run_policy_calls) == 1
        assert result["status"] in {"success", "error"}


class TestThePolicyIsBuiltOnceAndBeforeTheRecording:
    """One build per call, ahead of ``start_recording``, shared by every episode."""

    def test_the_build_precedes_the_recording_and_every_episode_gets_the_same_object(self) -> None:
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="mock", n_episodes=3, n_steps=4, dataset_root=ROOT)
        assert result["status"] in {"success", "error"}
        assert sim.order == ["start_recording", "run_policy", "run_policy", "run_policy", "stop_recording"]
        objects = [call["policy_object"] for call in sim.run_policy_calls]
        assert all(isinstance(obj, MockPolicy) for obj in objects)
        assert len({id(obj) for obj in objects}) == 1, "the policy was rebuilt per episode"

    def test_the_provider_and_config_are_still_forwarded(self) -> None:
        """The facade's report names the provider it was asked for, object or not."""
        sim = _OrderedSim()
        _run_tool(sim, policy_provider="mock", policy_config={}, n_episodes=1, n_steps=4, dataset_root=ROOT)
        call = sim.run_policy_calls[0]
        assert call["policy_provider"] == "mock"
        assert call["policy_config"] == {}

    def test_the_recording_less_path_forwards_no_policy_object(self) -> None:
        """Without a recording nothing is at stake on disk; the forwarded call is unchanged."""
        sim = _OrderedSim()
        _run_tool(sim, policy_provider="mock", n_episodes=2, n_steps=4)
        for call in sim.run_policy_calls:
            assert "policy_object" not in call
        assert sim.order == ["run_policy", "run_policy"]

    def test_an_unknown_provider_without_a_recording_is_the_facades_to_report(self) -> None:
        """Byte-identical to before: the facade answers, per episode, as it always did."""
        sim = _OrderedSim()
        result = _run_tool(sim, policy_provider="no_such_provider_xyz", n_episodes=1, n_steps=4)
        assert len(sim.run_policy_calls) == 1
        assert sim.run_policy_calls[0]["policy_provider"] == "no_such_provider_xyz"
        assert result["status"] == "success"  # the fake facade accepts anything


class _HistoryKeepingPolicy(MockPolicy):
    """A policy whose state would carry across episodes unless it is reset (flux3_action, groot, RTC)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset_seeds: list[int | None] = []
        self.raise_on_reset: Exception | None = None

    def reset(self, seed: int | None = None) -> None:
        if self.raise_on_reset is not None:
            raise self.raise_on_reset
        self.reset_seeds.append(seed)
        super().reset(seed=seed)


class TestTheSharedPolicyStartsEveryEpisodeFresh:
    """The one built object is handed to every episode, so the tool resets it between them.

    ``PolicyRunner.run`` resets a policy only when a seed was given, and this
    tool's default is ``seed=None``: without the reset an unseeded recording
    conditioned episode N+1 on episode N's history while the scene had jumped
    back to rest, and the drifted actions were what the dataset kept. Mirrors
    the facade's between-episode reset (``SimEngine.run_policy``).
    """

    @pytest.fixture
    def stateful(self, monkeypatch: pytest.MonkeyPatch) -> _HistoryKeepingPolicy:
        policy = _HistoryKeepingPolicy()
        monkeypatch.setattr(rp_mod, "_build_policy_before_recording", lambda *a, **k: policy)
        return policy

    def test_an_unseeded_recording_resets_the_policy_between_episodes(self, stateful) -> None:
        sim = _OrderedSim()
        _run_tool(sim, policy_provider="mock", n_episodes=3, n_steps=4, dataset_root=ROOT)
        assert len(sim.run_policy_calls) == 3
        assert stateful.reset_seeds == [None, None], "episodes 2 and 3 must start from a reset policy"

    def test_a_seeded_recording_resets_with_the_episodes_seed(self, stateful) -> None:
        sim = _OrderedSim()
        _run_tool(sim, policy_provider="mock", n_episodes=3, n_steps=4, seed=10, dataset_root=ROOT)
        assert stateful.reset_seeds == [11, 12]

    def test_one_episode_needs_no_reset(self, stateful) -> None:
        _run_tool(_OrderedSim(), policy_provider="mock", n_episodes=1, n_steps=4, dataset_root=ROOT)
        assert stateful.reset_seeds == []

    def test_a_reset_that_raises_is_logged_and_the_recording_goes_on(self, stateful, caplog) -> None:
        stateful.raise_on_reset = RuntimeError("no reset today")
        sim = _OrderedSim()
        with caplog.at_level("WARNING", logger=rp_mod.logger.name):
            _run_tool(sim, policy_provider="mock", n_episodes=2, n_steps=4, dataset_root=ROOT)
        assert len(sim.run_policy_calls) == 2
        assert any("no reset today" in r.getMessage() and "reset" in r.getMessage() for r in caplog.records)

    def test_the_recording_less_path_resets_nothing_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a recording no object is shared; each episode builds its own policy in the facade."""
        sim = _OrderedSim()
        _run_tool(sim, policy_provider="mock", n_episodes=2, n_steps=4)
        assert all("policy_object" not in call for call in sim.run_policy_calls)


class TestTheFirstEpisodeErrorReachesTheSummaryLine:
    """An agent reads the first content block; the reason must be in it."""

    def test_the_first_failed_episode_is_quoted(self) -> None:
        class _FailingSim(_FakeSim):
            def run_policy(self, **kwargs: Any) -> dict[str, Any]:
                self.run_policy_calls.append(kwargs)
                if len(self.run_policy_calls) == 2:
                    return {"status": "error", "content": [{"text": "run_policy: the robot has not moved"}]}
                return _ok_rollout("fine")

        result = _run_tool(_FailingSim(), n_episodes=3, n_steps=4)
        assert result["status"] == "error"
        assert _text(result).startswith("run_policy: 2/3 episodes ok")
        assert "first error (episode 2): run_policy: the robot has not moved" in _text(result)

    def test_a_clean_run_has_no_error_clause(self) -> None:
        result = _run_tool(_FakeSim(), n_episodes=2, n_steps=4)
        assert result["status"] == "success"
        assert "first error" not in _text(result)


def _truth(root: Path) -> tuple[int | None, int | None]:
    truth = rp_mod._read_parquet_truth(root)
    return truth.get("total_episodes"), truth.get("total_frames")


class TestAgainstARealDataset:
    """The measured defect, end to end: two recorded episodes survive a refused call."""

    @pytest.fixture
    def recorded_root(self, tmp_path: Path):
        pytest.importorskip("mujoco")
        pytest.importorskip("lerobot")
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation.mujoco.simulation import Simulation

        arm = tmp_path / "arm.xml"
        from tests.tools.test_run_policy_rate_agreement_preflight import _ARM_XML

        arm.write_text(_ARM_XML, encoding="utf-8")
        sim = Simulation()
        sim.create_world()
        assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
        root = tmp_path / "dataset"
        try:
            first = _run_tool(
                sim,
                robot_name="arm",
                policy_provider="mock",
                n_episodes=2,
                n_steps=5,
                control_frequency=30.0,
                dataset_fps=30,
                dataset_root=str(root),
            )
            assert first["status"] == "success", _text(first)
            assert _truth(root) == (2, 10), "premise: two episodes of five frames were recorded"
            yield sim, root
        finally:
            sim.cleanup()

    def test_an_unknown_provider_leaves_the_two_episodes(self, recorded_root) -> None:
        sim, root = recorded_root
        result = _run_tool(
            sim,
            robot_name="arm",
            policy_provider="no_such_provider_xyz",
            n_episodes=1,
            n_steps=5,
            control_frequency=30.0,
            dataset_fps=30,
            dataset_root=str(root),
        )
        assert result["status"] == "error"
        assert "no_such_provider_xyz" in _text(result)
        assert _truth(root) == (2, 10), "the refused call replaced the dataset at dataset_root"

    def test_a_checkpoint_that_does_not_exist_leaves_the_two_episodes(
        self, recorded_root, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sim, root = recorded_root

        def hub_miss(provider: str, **kwargs: Any) -> Any:
            raise FileNotFoundError("config.json not found on the HuggingFace Hub: nobody/does_not_exist_xyz")

        monkeypatch.setattr(policies_pkg, "create_policy", hub_miss)
        result = _run_tool(
            sim,
            robot_name="arm",
            policy_provider="lerobot_local",
            policy_config={"pretrained_name_or_path": "nobody/does_not_exist_xyz"},
            n_episodes=1,
            n_steps=5,
            control_frequency=30.0,
            dataset_fps=30,
            dataset_root=str(root),
        )
        assert result["status"] == "error"
        assert "nobody/does_not_exist_xyz" in _text(result)
        assert _truth(root) == (2, 10), "the refused call replaced the dataset at dataset_root"
