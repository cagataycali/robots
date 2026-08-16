# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Every pre-flight refusal on the two eval facades is returned, not just wired.

``run_policy`` / ``eval_policy`` / ``evaluate_benchmark`` each open with a block
of pre-flight guards, and the comment above the last one in ``evaluate_benchmark``
states what the block is for: the value is "checked before any policy is built so
a rate disagreement costs no weight download and no frame".

The reference facade pins all twelve of its refusals. The two eval facades did
not: six of their guard refusals had never been returned by any test, so what was
verified was that the call is *wired*, never that it *refuses*.

* ``evaluate_benchmark`` - ``video``, ``policy_config``, ``policy_kwargs`` and
  the dataset-rate check (4 of its 9 refusals).
* ``eval_policy`` - ``policy_kwargs`` and the dataset-rate check (2 of 11).

The rate check was pinned *structurally* instead, by an AST test asserting each
entry point calls ``_validate_recording_rate``, justified as covering "a driver
that needs a checkpoint, a benchmark registration or a live background thread to
reach". That justification does not hold for the refusal path: ``evaluate_benchmark``
runs its parameter guards before ``get_benchmark`` and before ``create_policy``, so
a refusal needs neither a registered benchmark nor a policy - which is what lets
every test below drive the real facade with no checkpoint and, for
``evaluate_benchmark``, no world at all.

Each refusal is the shared rule's verdict verbatim (envelope equality, not a
substring), because a facade that re-words a shared domain locally is how two
surfaces drift on what the same value means.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

# A two-actuator arm needing no asset download: enough for eval_policy's robot
# resolution, which runs before its parameter guards.
_ARM = """<mujoco><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""

# Values no recorder can honor: a key the config has no slot for, a frame rate
# that cannot be written, and a non-mapping where a mapping is documented.
UNHONORABLE_VIDEO: list[Any] = [
    {"filename": "/tmp/probe.mp4"},
    {"path": "/tmp/probe.mp4", "resolution": [320, 240]},
    {"path": "/tmp/probe.mp4", "fps": 0},
    "/tmp/probe.mp4",
]

# ``policy_config`` / ``policy_kwargs`` are splatted into ``create_policy`` and
# ``policy.get_actions``, so a non-mapping is a ``TypeError`` past the envelope.
NON_MAPPINGS: list[Any] = ["host=1", [1], 3]


def _text(result: dict[str, Any]) -> str:
    """The human-readable half of an agent-tool envelope."""
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


@pytest.fixture
def engine():
    """A bare engine: ``evaluate_benchmark``'s parameter guards need no world."""
    sim = MuJoCoSimEngine(tool_name="preflight_sim", mesh=False)
    yield sim
    sim.cleanup()


@pytest.fixture
def engine_with_arm(tmp_path):
    """``eval_policy`` resolves the robot before its parameter guards."""
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    sim = MuJoCoSimEngine(tool_name="preflight_arm_sim", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(xml))
    yield sim
    sim.cleanup()


def _benchmark_eval(sim, **kwargs: Any) -> dict[str, Any]:
    """Drive ``evaluate_benchmark`` with a name no benchmark is registered under.

    The parameter guards run before ``get_benchmark``, so a refusal that names
    the parameter (rather than the missing benchmark) is itself the evidence
    that the guard ran first.
    """
    return sim.evaluate_benchmark(benchmark_name="no_such_benchmark", robot_name="arm", n_episodes=1, **kwargs)


class TestEvaluateBenchmarkRefusesAVideoConfigItCannotHonor:
    """``video`` is a free-form dict, so a mistyped key has no signature to hit."""

    @pytest.mark.parametrize("video", UNHONORABLE_VIDEO, ids=["unknown-key", "resolution", "fps-0", "not-a-mapping"])
    def test_an_unhonorable_video_config_is_refused(self, engine, video: Any) -> None:
        result = _benchmark_eval(engine, video=video)
        assert result["status"] == "error"
        assert "video" in _text(result)

    @pytest.mark.parametrize("video", UNHONORABLE_VIDEO, ids=["unknown-key", "resolution", "fps-0", "not-a-mapping"])
    def test_the_refusal_is_the_shared_rules_verdict_verbatim(self, engine, video: Any) -> None:
        """A locally re-worded copy is how two surfaces drift on one value."""
        assert _benchmark_eval(engine, video=video) == engine._validate_video_config(video, "evaluate_benchmark")

    def test_the_refusal_names_the_parameter_and_not_the_missing_benchmark(self, engine) -> None:
        """Proves the guard precedes ``get_benchmark``, as its own comment claims."""
        text = _text(_benchmark_eval(engine, video={"filename": "/tmp/probe.mp4"}))
        assert "unknown key 'filename'" in text
        assert "no_such_benchmark" not in text


class TestTheEvalFacadesRefuseANonMappingPolicyMapping:
    """Both mappings are splatted downstream, so neither may be a non-mapping."""

    @pytest.mark.parametrize("param", ["policy_config", "policy_kwargs"])
    @pytest.mark.parametrize("value", NON_MAPPINGS, ids=["str", "list", "int"])
    def test_evaluate_benchmark_refuses_it(self, engine, param: str, value: Any) -> None:
        result = _benchmark_eval(engine, **{param: value})
        assert result["status"] == "error"
        assert result == engine._validate_policy_mapping(value, param, "evaluate_benchmark")

    @pytest.mark.parametrize("value", NON_MAPPINGS, ids=["str", "list", "int"])
    def test_eval_policy_refuses_a_non_mapping_policy_kwargs(self, engine_with_arm, value: Any) -> None:
        result = engine_with_arm.eval_policy(
            robot_name="arm", policy_provider="mock", n_episodes=1, max_steps=2, policy_kwargs=value
        )
        assert result["status"] == "error"
        assert result == engine_with_arm._validate_policy_mapping(value, "policy_kwargs", "eval_policy")


class TestTheEvalFacadesRefuseADatasetRateDisagreement:
    """One frame per control step and no decimation: a differing fps only mislabels."""

    def _record_at(self, sim, tmp_path, fps: int, name: str) -> None:
        pytest.importorskip("lerobot")
        result = sim.start_recording(repo_id=f"local/{name}", task="hold", fps=fps, root=str(tmp_path / name))
        assert result["status"] == "success"

    def test_evaluate_benchmark_refuses_a_rate_the_open_dataset_cannot_carry(self, engine_with_arm, tmp_path) -> None:
        self._record_at(engine_with_arm, tmp_path, 30, "bench_ds")
        result = _benchmark_eval(engine_with_arm, control_frequency=50.0)
        assert result["status"] == "error"
        assert result == engine_with_arm._validate_recording_rate(50.0, "evaluate_benchmark")

    def test_eval_policy_refuses_a_rate_the_open_dataset_cannot_carry(self, engine_with_arm, tmp_path) -> None:
        self._record_at(engine_with_arm, tmp_path, 30, "eval_ds")
        result = engine_with_arm.eval_policy(
            robot_name="arm", policy_provider="mock", n_episodes=1, max_steps=2, control_frequency=50.0
        )
        assert result["status"] == "error"
        assert result == engine_with_arm._validate_recording_rate(50.0, "eval_policy")

    def test_a_matching_rate_is_not_refused(self, engine_with_arm, tmp_path) -> None:
        """Without this, refusing every rate would satisfy the tests above."""
        self._record_at(engine_with_arm, tmp_path, 30, "match_ds")
        text = _text(_benchmark_eval(engine_with_arm, control_frequency=30.0))
        assert "declares 30 fps" not in text
        assert "no benchmark registered" in text  # fell through to the lookup


class TestARefusedPreflightCostsNothing:
    """What the guard block exists for: no weight download and no frame."""

    def test_no_policy_is_built(self, engine, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.policies as policies

        built: list[str] = []
        monkeypatch.setattr(policies, "create_policy", lambda *a, **k: built.append("built"))
        assert _benchmark_eval(engine, policy_config="host=1")["status"] == "error"
        assert built == [], "a refused evaluation built a policy"

    def test_the_benchmark_is_never_resolved(self, engine, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation import benchmark as benchmark_mod

        looked_up: list[str] = []

        def _spy(name: str):
            looked_up.append(name)
            return None

        monkeypatch.setattr(benchmark_mod, "get_benchmark", _spy)
        assert _benchmark_eval(engine, policy_kwargs=[1])["status"] == "error"
        assert looked_up == [], "a refused evaluation resolved the benchmark"

    def test_no_frame_reaches_the_open_dataset(self, engine_with_arm, tmp_path) -> None:
        pytest.importorskip("lerobot")
        result = engine_with_arm.start_recording(
            repo_id="local/cost_ds", task="hold", fps=30, root=str(tmp_path / "cost_ds")
        )
        assert result["status"] == "success"
        recorder = engine_with_arm._active_recorder()
        assert recorder is not None
        before = recorder.frame_count
        assert _benchmark_eval(engine_with_arm, video={"filename": "/tmp/probe.mp4"})["status"] == "error"
        assert recorder.frame_count == before == 0


class TestOneRuleWordedOncePerSurface:
    """The two facades must differ from the reference only in the method named."""

    @pytest.mark.parametrize("video", UNHONORABLE_VIDEO, ids=["unknown-key", "resolution", "fps-0", "not-a-mapping"])
    def test_the_video_verdict_differs_from_run_policys_only_by_the_context(self, engine, video: Any) -> None:
        under_bench = _text(engine._validate_video_config(video, "evaluate_benchmark"))
        under_run = _text(engine._validate_video_config(video, "run_policy"))
        assert under_bench.removeprefix("evaluate_benchmark: ") == under_run.removeprefix("run_policy: ")
        assert under_bench.startswith("evaluate_benchmark: ")

    @pytest.mark.parametrize("param", ["policy_config", "policy_kwargs"])
    def test_the_mapping_verdict_differs_from_run_policys_only_by_the_context(self, engine, param: str) -> None:
        under_eval = _text(engine._validate_policy_mapping("host=1", param, "eval_policy"))
        under_run = _text(engine._validate_policy_mapping("host=1", param, "run_policy"))
        assert under_eval.removeprefix("eval_policy: ") == under_run.removeprefix("run_policy: ")
