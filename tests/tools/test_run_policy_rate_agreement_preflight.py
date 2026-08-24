# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``run_policy`` tool's own two rates must agree before it opens a recording.

The tool owns both halves of one rule. ``dataset_fps`` becomes the
LeRobotDataset's declared frame rate and ``control_frequency`` is the rate the
recorder is actually driven at - one frame per control step, never decimated -
so LeRobot's positional timestamps (``timestamp = frame_index / fps``) describe
the episode only when the two are EQUAL.

Each rate was already checked on its own domain: ``control_frequency`` by the
tool's pre-flight, ``dataset_fps`` by ``start_recording`` before it touches the
target directory. Their *equality* was left to the rollout entry point - which
this tool reaches only inside the episode loop, after
``start_recording(overwrite=True)`` has replaced whatever was at
``dataset_root``. Measured over a real MuJoCo rollout that had recorded one
episode of five frames, ``dataset_fps=30`` with ``control_frequency=50.0``::

    meta/info.json: total_episodes=1, total_frames=5   (before the call)
    meta/info.json: total_episodes=0, total_frames=0   (after it)
    run_policy: 0/2 episodes ok | parquet-truth: total_episodes=0, total_frames=0

Neither argument was wrong on its own, so nothing refused the pair; the caller
lost the dataset and recorded nothing in its place. The tool's own pre-flight
block states the principle for every other knob it forwards - ``seed``
("reached NumPy inside the loop after step 2 had already created a dataset"),
``video``, the provider keyword bags ("otherwise the tool leaves an empty
dataset behind and reports every episode as raised") and ``stop_when``. The
per-knob structural sweep in
``tests/tools/test_run_policy_rollout_knob_preflight.py`` could not see this
one: it derives the knobs forwarded to ``simulation.run_policy(...)``, and a
rule between two parameters is not a property of either.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.run_policy as rp_mod
from strands_robots.simulation.recording import (
    dataset_recording_option_error,
    rate_mismatch_explanation,
    requested_rate_mismatch_reason,
)
from strands_robots.utils import positive_finite_number_error
from tests.tools.test_run_policy import _FakeSim

#: Rates that disagree in both directions, plus the pair the library defaults
#: used to collide at (``fps=30`` against ``control_frequency=50.0``).
DISAGREEING: list[tuple[int, float]] = [(30, 50.0), (50, 30.0), (60, 50.0), (30, 33.3), (25, 30.0)]

#: Pairs a caller may legitimately pass: equal rates, including the integral
#: and NumPy spellings a value read out of a config arrives as.
AGREEING: list[tuple[Any, Any]] = [(30, 30.0), (50, 50.0), (30, 30), (25, 25.0), (30, 30.0000000001)]


def _run_tool(sim: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the tool with kwargs mypy cannot narrow.

    Several values are deliberately outside the declared parameter types - that
    is the point of the test - so they go through one funnel rather than being
    suppressed at every call site.
    """
    return dict(rp_mod.run_policy(sim, **kwargs))


def _text(result: dict[str, Any]) -> str:
    return str((result.get("content") or [{}])[0].get("text", ""))


def _truth(root: Path) -> tuple[int | None, int | None]:
    """Read ``(total_episodes, total_frames)`` from a dataset's ``meta/info.json``."""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return info.get("total_episodes"), info.get("total_frames")


# One actuated hinge plus a camera: enough for a rollout to drive something and
# for the dataset schema to declare an image column, with no asset download.
_ARM_XML = """
<mujoco model="rate_agreement_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="0 -1 0.4" xyaxes="1 0 0 0 0.3 1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def recording_sim(tmp_path):
    """A live MuJoCo world that can write a real LeRobotDataset."""
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation.mujoco.simulation import Simulation

    arm = tmp_path / "arm.xml"
    arm.write_text(_ARM_XML, encoding="utf-8")
    sim = Simulation()
    sim.create_world()
    assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
    try:
        yield sim
    finally:
        sim.cleanup()


class TestTheRefusalPrecedesTheDestruction:
    """The reason this is a defect and not a tidiness question.

    ``start_recording`` is forwarded ``overwrite=True``, so reaching it with a
    pair the episode loop will refuse costs the caller the dataset already at
    that root - and leaves an empty one in its place.
    """

    @staticmethod
    def _record_one_episode(sim, root: Path, fps: int) -> tuple[int | None, int | None]:
        """Record a real episode at ``fps`` and return its parquet truth."""
        assert (
            sim.start_recording(repo_id="local/rate_agreement", task="t", fps=fps, root=str(root), overwrite=True)[
                "status"
            ]
            == "success"
        )
        assert (
            sim.run_policy(
                robot_name="arm", policy_provider="mock", n_steps=5, control_frequency=float(fps), fast_mode=True
            )["status"]
            == "success"
        )
        assert sim.stop_recording()["status"] == "success"
        return _truth(root)

    def test_a_disagreeing_pair_leaves_an_existing_dataset_untouched(self, recording_sim, tmp_path) -> None:
        root = tmp_path / "dataset"
        before = self._record_one_episode(recording_sim, root, 30)
        assert before == (1, 5), f"premise: the fixture recorded an episode, got {before}"

        result = _run_tool(
            recording_sim,
            robot_name="arm",
            policy_provider="mock",
            n_episodes=2,
            n_steps=5,
            control_frequency=50.0,
            dataset_fps=30,
            dataset_root=str(root),
            dataset_repo_id="local/rate_agreement",
        )
        assert result["status"] == "error"
        assert "30 fps" in _text(result) and "control_frequency=50 Hz" in _text(result)
        assert _truth(root) == before, "the refused call replaced the dataset at dataset_root"

    def test_the_preserved_dataset_still_reopens_with_its_frames(self, recording_sim, tmp_path) -> None:
        """Round trip: the episode survives as a dataset, not just as metadata."""
        root = tmp_path / "dataset"
        self._record_one_episode(recording_sim, root, 30)
        _run_tool(
            recording_sim,
            robot_name="arm",
            policy_provider="mock",
            n_episodes=2,
            n_steps=5,
            control_frequency=50.0,
            dataset_fps=30,
            dataset_root=str(root),
            dataset_repo_id="local/rate_agreement",
        )
        # Asserted before reopening: a LeRobotDataset whose root has been
        # emptied falls back to the Hub, so a regression must fail on the local
        # metadata rather than on a network lookup for a repo that is local-only.
        assert _truth(root) == (1, 5)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local/rate_agreement", root=str(root))
        assert dataset.fps == 30
        assert dataset.num_frames == 5
        assert list(root.rglob("*.mp4"))

    def test_an_agreeing_pair_records_the_episodes_it_was_asked_for(self, recording_sim, tmp_path) -> None:
        """Over-reach control, end to end: the honorable pair still records."""
        root = tmp_path / "dataset"
        result = _run_tool(
            recording_sim,
            robot_name="arm",
            policy_provider="mock",
            n_episodes=2,
            n_steps=5,
            control_frequency=30.0,
            dataset_fps=30,
            dataset_root=str(root),
            dataset_repo_id="local/rate_agreement",
        )
        assert result["status"] == "success", _text(result)
        assert _truth(root) == (2, 10)


class TestTheRequestedRatesMustAgree:
    """A pair that cannot describe one episode is refused, naming both rates."""

    @pytest.mark.parametrize(("fps", "rate"), DISAGREEING, ids=repr)
    def test_a_disagreeing_pair_is_refused(self, fps: int, rate: float) -> None:
        result = _run_tool(
            _FakeSim(),
            n_episodes=2,
            n_steps=4,
            control_frequency=rate,
            dataset_fps=fps,
            dataset_root="/tmp/run-policy-rate-agreement",
        )
        assert result["status"] == "error"
        assert f"{fps} fps" in _text(result)
        assert f"control_frequency={rate:g}" in _text(result)

    def test_the_refusal_carries_the_shared_account_of_the_distortion(self) -> None:
        """One disagreement, so one explanation - not a second account of it."""
        result = _run_tool(
            _FakeSim(),
            n_episodes=1,
            n_steps=4,
            control_frequency=50.0,
            dataset_fps=30,
            dataset_root="/tmp/run-policy-rate-agreement",
        )
        assert rate_mismatch_explanation(30, 50.0) in _text(result)

    def test_the_refusal_names_both_remedies_in_the_callers_own_spelling(self) -> None:
        """The advised argument is one the caller of *this tool* can type."""
        result = _run_tool(
            _FakeSim(),
            n_episodes=1,
            n_steps=4,
            control_frequency=50.0,
            dataset_fps=30,
            dataset_root="/tmp/run-policy-rate-agreement",
        )
        assert "pass control_frequency=30" in _text(result)
        assert "dataset_fps=50" in _text(result)
        assert "fps=50)" not in _text(result).replace("dataset_fps=50)", "")

    def test_nothing_is_started_and_no_episode_is_recorded(self) -> None:
        sim = _FakeSim()
        result = _run_tool(
            sim,
            n_episodes=3,
            n_steps=4,
            control_frequency=50.0,
            dataset_fps=30,
            dataset_root="/tmp/run-policy-rate-agreement",
        )
        assert result["status"] == "error"
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []
        assert sim.stop_recording_calls == []
        payload = next((b["json"] for b in result.get("content") or [] if "json" in b), None)
        assert payload is None, "a refused pre-flight reports no episode records"


class TestUsablePairsStillRun:
    """Over-reach control: nothing a caller could legitimately pass is refused."""

    @pytest.mark.parametrize(("fps", "rate"), AGREEING, ids=repr)
    def test_an_agreeing_pair_drives_the_loop_and_records(self, fps: Any, rate: Any) -> None:
        sim = _FakeSim()
        result = _run_tool(
            sim,
            n_episodes=2,
            n_steps=4,
            control_frequency=rate,
            dataset_fps=fps,
            dataset_root="/tmp/run-policy-rate-agreement",
        )
        assert len(sim.start_recording_calls) == 1
        assert sim.start_recording_calls[0]["fps"] == fps
        assert len(sim.run_policy_calls) == 2
        # ``status`` is "error" only because the fake writes no parquet truth.
        assert "2/2 episodes ok" in _text(result)

    @pytest.mark.parametrize("rate", [50.0, 12.5, 1.0], ids=repr)
    def test_a_recordingless_rollout_runs_at_any_rate(self, rate: float) -> None:
        """``dataset_fps`` is forwarded nowhere without ``dataset_root``.

        The default ``dataset_fps=30`` disagrees with most rates a caller picks,
        so gating the new check on a requested recording is what keeps the
        smoke-test path (documented as "the loop runs without recording") open.
        """
        sim = _FakeSim()
        result = _run_tool(sim, n_episodes=2, n_steps=4, control_frequency=rate)
        assert result["status"] == "success"
        assert len(sim.run_policy_calls) == 2
        assert sim.start_recording_calls == []

    def test_the_documented_defaults_agree_with_each_other(self) -> None:
        """A caller who supplies neither rate must not be refused.

        Read off the shipped signature rather than restated, so a default moved
        to a colliding value fails here instead of in the field.
        """
        fn = _tool_function(_tool_source())
        defaults = {
            arg.arg: ast.literal_eval(default)
            for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
            if default is not None
        }
        assert defaults["control_frequency"] == 30.0
        assert defaults["dataset_fps"] == 30
        assert (
            requested_rate_mismatch_reason("run_policy", defaults["dataset_fps"], defaults["control_frequency"]) is None
        )


class TestTheHelperIsRobust:
    """The reason helper's own domain, driven directly."""

    @pytest.mark.parametrize(("fps", "rate"), AGREEING, ids=repr)
    def test_agreeing_rates_return_none(self, fps: Any, rate: Any) -> None:
        assert requested_rate_mismatch_reason("run_policy", fps, rate) is None

    @pytest.mark.parametrize("fps", [0, -30, 2.5, float("nan"), True, np.bool_(True), "30", None, [30]], ids=repr)
    def test_an_fps_outside_the_writable_domain_is_left_to_its_own_guard(self, fps: Any) -> None:
        """Premise: each of these IS refused by the fps guard, as that error."""
        assert dataset_recording_option_error("start_recording", fps) is not None
        assert requested_rate_mismatch_reason("run_policy", fps, 50.0) is None

    @pytest.mark.parametrize(
        "rate", [0.0, -5.0, float("nan"), float("inf"), True, np.bool_(True), "30", None, [30]], ids=repr
    )
    def test_a_rate_outside_its_domain_is_left_to_its_own_guard(self, rate: Any) -> None:
        """Each of these is its own parameter's error, not a rate disagreement."""
        assert requested_rate_mismatch_reason("run_policy", 30, rate) is None

    @pytest.mark.parametrize("value", [True, np.bool_(True)], ids=repr)
    def test_a_boolean_rate_is_not_read_as_one_hertz(self, value: Any) -> None:
        """``bool`` IS a ``numbers.Real``, so the type check alone lets it through.

        ``float(True)`` is ``1.0``, and 1 is a rate this helper does compare (the
        premise below), so a boolean would have been diagnosed as a genuine 1 Hz
        rollout rather than reported as the flag-shaped mistake it is. That is
        what the shared ``is_boolean`` predicate answers; ``numpy.bool_`` is
        covered twice over, since it is not a ``numbers.Real`` either.
        """
        assert requested_rate_mismatch_reason("run_policy", 30, 1.0) is not None
        assert requested_rate_mismatch_reason("run_policy", 30, value) is None
        assert requested_rate_mismatch_reason("run_policy", value, 30.0) is None

    def test_a_fractional_capture_rate_offers_only_the_remedy_it_can(self) -> None:
        """No whole ``fps`` describes 33.3 Hz, so only the rate remedy is advised."""
        reason = requested_rate_mismatch_reason("run_policy", 30, 33.3)
        assert reason is not None
        assert "pass control_frequency=30" in reason
        assert "record at the rollout's rate" not in reason

    def test_the_message_is_prefixed_with_the_calling_surface(self) -> None:
        reason = requested_rate_mismatch_reason("some_surface", 30, 50.0)
        assert reason is not None
        assert reason.startswith("some_surface: ")

    def test_the_fps_argument_is_named_as_the_caller_spells_it(self) -> None:
        default = requested_rate_mismatch_reason("start_recording", 30, 50.0)
        renamed = requested_rate_mismatch_reason("run_policy", 30, 50.0, fps_param="dataset_fps")
        assert default is not None and renamed is not None
        assert "(fps=50)" in default
        assert "(dataset_fps=50)" in renamed


#: Spellings a rate arrives as - the usable ones, the NumPy scalars a value read
#: out of a config or produced by a comparison carries, and the unusable ones.
#: ``numpy.int64`` and ``numpy.float32`` are neither ``int`` nor ``float``
#: subclasses, which is what makes the parity below load-bearing.
RATE_SPELLINGS: list[Any] = [
    30,
    30.0,
    50.0,
    np.int64(50),
    np.float32(30.0),
    np.float64(50.0),
    0,
    -30,
    2.5,
    float("nan"),
    float("inf"),
    True,
    np.bool_(True),
    "30",
    None,
]


class TestTheGuardJudgesExactlyThePairsBothDomainsAccept:
    """Derived parity: it may neither duplicate a domain nor decline to judge.

    This guard runs *before* either rate has been through its own domain, so its
    accepted set has to be derived from those two domains rather than guessed.
    Too narrow and it silently passes a colliding pair (an ``isinstance(int |
    float)`` narrowing declines every NumPy scalar); too wide and it reports a
    rate disagreement for what is really one parameter's own error.
    """

    @pytest.mark.parametrize("rate", RATE_SPELLINGS, ids=repr)
    @pytest.mark.parametrize("fps", RATE_SPELLINGS, ids=repr)
    def test_the_verdict_follows_the_two_domains(self, fps: Any, rate: Any) -> None:
        judged = requested_rate_mismatch_reason("run_policy", fps, rate) is not None
        both_usable = (
            dataset_recording_option_error("start_recording", fps) is None
            and positive_finite_number_error(rate, "control_frequency", "run_policy") is None
        )
        if not both_usable:
            assert not judged, "a pair one of whose halves is its own parameter error was read as a rate disagreement"
            return
        if float(fps) == float(rate):
            assert not judged
        else:
            assert judged, "a pair both domains accept was passed through unjudged"

    def test_the_grid_reaches_all_three_outcomes(self) -> None:
        """Non-vacuity: a grid that never lands in a bucket asserts nothing there."""
        outcomes = set()
        for fps in RATE_SPELLINGS:
            for rate in RATE_SPELLINGS:
                usable = (
                    dataset_recording_option_error("start_recording", fps) is None
                    and positive_finite_number_error(rate, "control_frequency", "run_policy") is None
                )
                if not usable:
                    outcomes.add("one half unusable")
                elif float(fps) == float(rate):
                    outcomes.add("agreeing")
                else:
                    outcomes.add("disagreeing")
        assert outcomes == {"one half unusable", "agreeing", "disagreeing"}


# --------------------------------------------------------------------------
# Structural: the pair is judged before the statement that opens the recording
# --------------------------------------------------------------------------


def _tool_source() -> str:
    return Path(inspect.getfile(rp_mod)).read_text(encoding="utf-8")


def _tool_function(source: str) -> ast.FunctionDef:
    """Return the ``run_policy`` tool's own FunctionDef from ``source``."""
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_policy"
        and any(arg.arg == "simulation" for arg in node.args.args)
    )


def _pairs_judged_before_the_recording(source: str) -> list[set[str]]:
    """Parameter-name sets read together by a call before ``start_recording``.

    One entry per call reached in the pre-flight region, holding the tool
    parameters named among its arguments - so a rule *between* parameters shows
    up as a set of more than one, which a per-knob scan cannot represent.
    """
    fn = _tool_function(source)
    params = {arg.arg for arg in fn.args.args + fn.args.kwonlyargs}
    # Enumerate from 1: ``fn.body[0]`` is the docstring, and it documents the
    # ``start_recording(root=dataset_root, ...)`` call by name.
    start_index = next(
        index
        for index, node in enumerate(fn.body)
        if index > 0 and "start_recording(" in (ast.get_source_segment(source, node) or "")
    )
    judged: list[set[str]] = []
    for statement in fn.body[1:start_index]:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            named = {
                child.id
                for argument in list(node.args) + [keyword.value for keyword in node.keywords]
                for child in ast.walk(argument)
                if isinstance(child, ast.Name) and child.id in params
            }
            if named:
                judged.append(named)
    return judged


class TestThePairIsJudgedBeforeTheRecordingOpens:
    """ "The facade will refuse it" is not enough when the facade runs after the wipe."""

    def test_the_two_rates_are_read_together_before_step_two(self) -> None:
        judged = _pairs_judged_before_the_recording(_tool_source())
        assert any({"dataset_fps", "control_frequency"} <= names for names in judged), (
            f"no pre-flight call reads both rates; pairs judged: {[sorted(n) for n in judged]}"
        )

    def test_the_scan_catches_a_removed_guard(self) -> None:
        """Planted defect: a clean result must mean a clean source, not a blind scan."""
        source = _tool_source()
        guard = (
            "        if rate_error := requested_rate_mismatch_reason(\n"
            '            "run_policy", dataset_fps, control_frequency, fps_param="dataset_fps"\n'
            "        ):\n"
            "            return _err(rate_error)\n"
        )
        assert source.count(guard) == 1
        planted = source.replace(guard, "")
        judged = _pairs_judged_before_the_recording(planted)
        assert not any({"dataset_fps", "control_frequency"} <= names for names in judged)
