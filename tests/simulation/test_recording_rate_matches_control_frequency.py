# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A dataset's declared frame rate must be the rate its frames were captured at.

``start_recording(fps=...)`` fixes the rate written into the LeRobotDataset
metadata, and LeRobot derives every timestamp from it positionally
(``timestamp = frame_index / fps``). The dataset recorder is driven once per
control step with **no decimation**, so the rate frames are really captured at
is the rollout's ``control_frequency`` - a differing ``fps`` cannot be honored,
only mislabelled.

The two library defaults were exactly such a pair (``fps=30`` against
``control_frequency=50.0``), which is what the documented record-then-rollout
sequence in ``docs/recording.md`` used. Measured on a position-servo arm before
the guard::

    fps=30 cf=50.0 -> captured 0.0200s/frame, timestamped 0.0333s/frame (1.667x)
    fps=50 cf=50.0 -> captured 0.0200s/frame, timestamped 0.0200s/frame (1.000x)

with ``start_recording``, the rollout and ``stop_recording`` all reporting
``status="success"`` and no log line. The distortion is the control period a
policy trains on, and ``replay_episode`` derives its per-frame physics budget
from the dataset rate on the invariant that "the recorded control frequency IS
the dataset fps" - so the same episode also replays at the wrong speed
(round-tripped to 0.0000 rad at matching rates, 0.0317 rad at the defaults).

Refused rather than warned, matching the sibling rate guard in the same module:
``_verify_resume_schema`` already refuses an ``fps`` that disagrees with the
dataset on disk. The refusal lands before any frame is written, so a caller
loses nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402
from strands_robots.simulation.recording import (  # noqa: E402
    dataset_rate_mismatch_error,
    dataset_rate_mismatch_reason,
    recorder_dataset_fps,
)

_ARM = """<mujoco><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
<body name="l2" pos="0.15 0 0">
<joint name="j2" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/></body></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/>
<position name="a2" joint="j2" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""


class _Hold(Policy):
    """State-only policy: no camera renders, so these tests stay fast."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "hold"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, keys) -> None:  # noqa: ANN001
        pass

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: ANN001, ANN003
        return [{k: 0.3 for k in self._keys}]


@pytest.fixture
def sim(tmp_path):
    """A two-actuator arm needing no asset download."""
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    engine = MuJoCoSimEngine(tool_name="rate_guard_sim", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(xml))
    yield engine
    engine.cleanup()


def _record(sim, tmp_path, fps: int, name: str = "ds") -> str:
    """Open a recording at ``fps``; each case gets its OWN root.

    ``start_recording`` RESUMES an existing dataset directory and inherits its
    rate from disk, so sharing one root across cases makes every case report
    the first one's fps - which looks exactly like ``fps`` being ignored.
    """
    root = tmp_path / name
    result = sim.start_recording(repo_id=f"local/{name}", task="hold", fps=fps, root=str(root))
    assert result["status"] == "success"
    return str(root)


def _frames_on_disk(root: str) -> int:
    parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
    if not parquets:
        return 0
    pd = pytest.importorskip("pandas")
    return sum(len(pd.read_parquet(p)) for p in parquets)


class TestTheLibraryDefaultsAreRefused:
    """The out-of-the-box pair is the mismatch, so it is the headline case."""

    def test_recording_at_30_then_rolling_at_50_is_refused(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert result["status"] == "error"

    def test_the_refusal_names_both_rates_and_the_distortion(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)["content"][0]["text"]
        assert "30 fps" in text
        assert "control_frequency=50" in text
        assert "1.667x" in text

    def test_the_refusal_names_both_remedies(self, sim, tmp_path):
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)["content"][0]["text"]
        assert "control_frequency=30" in text
        assert "start_recording(fps=50, overwrite=True)" in text

    def test_no_frame_is_written(self, sim, tmp_path):
        """The refusal precedes capture, so the caller loses no episode."""
        root = _record(sim, tmp_path, 30)
        sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert _frames_on_disk(root) == 0


class TestMatchingRatesAreUnaffected:
    def test_a_rollout_at_the_declared_rate_records(self, sim, tmp_path):
        root = _record(sim, tmp_path, 50)
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=50.0, n_steps=10)
        assert result["status"] == "success"
        sim.stop_recording()
        assert _frames_on_disk(root) == 10

    def test_the_episode_duration_is_not_distorted(self, sim, tmp_path):
        """The declared span must equal the captured span - the point of the guard."""
        pd = pytest.importorskip("pandas")
        root = _record(sim, tmp_path, 25)
        sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=25.0, n_steps=10)
        sim.stop_recording()
        parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
        stamps = pd.concat([pd.read_parquet(p) for p in parquets])["timestamp"].tolist()
        assert stamps[-1] - stamps[0] == pytest.approx((len(stamps) - 1) / 25.0, abs=1e-6)

    def test_a_rollout_with_no_recording_open_is_never_refused(self, sim):
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=17.0, n_steps=5)
        assert result["status"] == "success"


class TestTheAdvisedRemedyIsUsable:
    """A recommendation that would not work must not be printed."""

    def test_following_the_control_frequency_remedy_records_cleanly(self, sim, tmp_path):
        root = _record(sim, tmp_path, 30)
        refused = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), n_steps=10)
        assert refused["status"] == "error"
        # The message says: pass control_frequency=30. Do exactly that.
        result = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=30.0, n_steps=10)
        assert result["status"] == "success"
        sim.stop_recording()
        assert _frames_on_disk(root) == 10

    def test_a_fractional_capture_rate_offers_only_the_rate_it_can(self, sim, tmp_path):
        """``fps`` must be a whole number, so 33.3 Hz has no re-record remedy."""
        _record(sim, tmp_path, 30)
        text = sim.run_policy(robot_name="arm", policy_object=_Hold(["a1", "a2"]), control_frequency=33.3, n_steps=5)[
            "content"
        ][0]["text"]
        assert "control_frequency=30" in text
        assert "start_recording(fps=" not in text


class TestTheGuardHelperIsRobust:
    def test_matching_rates_return_none(self):
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(30), 30.0) is None

    def test_float_noise_below_the_tolerance_is_not_a_mismatch(self):
        """A rate carried as a float must not be refused for representation noise."""
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(30), 30.0 + 1e-12) is None

    def test_a_dataset_with_no_readable_rate_does_not_block(self):
        """An unexpected LeRobot layout must not refuse a valid rollout."""
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(None), 50.0) is None

    def test_a_fractional_dataset_rate_does_not_block(self):
        assert dataset_rate_mismatch_error("run_policy", _FakeRecorder(29.97), 50.0) is None

    def test_the_message_is_prefixed_with_the_calling_method(self):
        err = dataset_rate_mismatch_error("eval_policy", _FakeRecorder(30), 50.0)
        assert err is not None
        assert err["content"][0]["text"].startswith("eval_policy: ")

    @pytest.mark.parametrize("rate", [30, 30.0])
    def test_the_rate_is_read_from_either_lerobot_spelling(self, rate):
        assert recorder_dataset_fps(_FakeRecorder(rate)) == 30
        assert recorder_dataset_fps(_MetaOnlyRecorder(rate)) == 30

    def test_a_boolean_rate_is_not_read_as_one(self):
        assert recorder_dataset_fps(_FakeRecorder(True)) is None


class TestTheRunnerLayerCarriesItsOwnGuarantee:
    """``PolicyRunner`` is driven directly, with the engine's guard off the path.

    ``docs/policies/lerobot-local.md`` names ``PolicyRunner.run`` beside
    ``run_policy`` as a caller surface, and ``_control_substeps`` already raises
    for a bad ``control_substeps`` on the stated grounds that "the public entry
    points reject such a value ... this raise is the guarantee for callers
    driving ``PolicyRunner`` directly". The recording rate needs the same
    treatment: measured before the guard, a direct rollout at ``fps=30`` /
    ``control_frequency=50`` wrote 20 frames declaring 0.0333s each for a
    capture 0.0200s apart (1.667x) and reported ``status="success"``.
    """

    def _keys(self, sim) -> list[str]:  # noqa: ANN001
        return list(sim.robot_action_keys("arm"))

    def test_direct_run_refuses_a_rate_the_recording_cannot_describe(self, sim, tmp_path):
        root = _record(sim, tmp_path, fps=30)
        with pytest.raises(ValueError) as excinfo:
            PolicyRunner(sim).run(
                "arm",
                _Hold(self._keys(sim)),
                n_steps=20,
                control_frequency=50.0,
                on_frame=sim._make_run_policy_hook("arm", "hold"),
            )
        message = str(excinfo.value)
        assert message.startswith("PolicyRunner.run: ")
        assert "declares 30 fps" in message
        assert "control_frequency=50 Hz" in message
        # Refused before the recorder was touched, so nothing has to be undone.
        assert sim._active_recorder().frame_count == 0
        assert _frames_on_disk(root) == 0

    def test_direct_evaluate_refuses_the_same_disagreement(self, sim, tmp_path):
        """The eval loop writes into the same open recording, so it refuses too."""
        root = _record(sim, tmp_path, fps=30, name="eval_ds")
        with pytest.raises(ValueError) as excinfo:
            PolicyRunner(sim).evaluate(
                "arm",
                _Hold(self._keys(sim)),
                n_episodes=1,
                max_steps=20,
                control_frequency=50.0,
                on_frame=sim._make_run_policy_hook("arm", "hold"),
            )
        assert str(excinfo.value).startswith("PolicyRunner.evaluate: ")
        assert _frames_on_disk(root) == 0

    def test_a_matching_rate_still_records_an_exact_timebase(self, sim, tmp_path):
        """Control: the guard must not cost a correctly configured direct caller."""
        root = _record(sim, tmp_path, fps=50, name="aligned_ds")
        result = PolicyRunner(sim).run(
            "arm",
            _Hold(self._keys(sim)),
            n_steps=20,
            control_frequency=50.0,
            on_frame=sim._make_run_policy_hook("arm", "hold"),
        )
        assert result["status"] == "success"
        assert sim.stop_recording()["status"] == "success"
        assert _frames_on_disk(root) == 20
        pd = pytest.importorskip("pandas")
        parquets = [p for p in Path(root).rglob("*.parquet") if "data" in p.parts]
        stamps = pd.concat([pd.read_parquet(p) for p in parquets])["timestamp"].tolist()
        assert stamps[1] - stamps[0] == pytest.approx(1.0 / 50.0, abs=1e-9)

    def test_a_rollout_with_no_recording_open_is_unaffected(self, sim):
        """Control: the rates are only comparable while a recording is open."""
        result = PolicyRunner(sim).run("arm", _Hold(self._keys(sim)), n_steps=5, control_frequency=50.0)
        assert result["status"] == "success"

    def test_a_sim_without_the_recording_hooks_is_not_probed(self):
        """A backend that cannot record, or a test double, has neither hook."""
        PolicyRunner(object())._reject_recording_rate_mismatch(50.0, "PolicyRunner.run")

    def test_a_sim_reporting_no_active_recorder_is_not_probed(self):
        """``_is_recording`` and ``_active_recorder`` can disagree; trust neither alone."""

        class _Engine:
            def _is_recording(self) -> bool:
                return True

            def _active_recorder(self) -> None:
                return None

        PolicyRunner(_Engine())._reject_recording_rate_mismatch(50.0, "PolicyRunner.run")


@pytest.mark.parametrize(
    ("fps", "control_frequency"),
    [(30, 50.0), (50, 30.0), (30, 30.0), (50, 50.0), (25, 25.0), (None, 50.0), (29.97, 50.0)],
)
def test_the_runner_and_the_engine_refuse_the_same_rates(fps, control_frequency):
    """One rule, two surfaces: the verdict and the reason must not diverge.

    The engine reports through a tool envelope and the runner raises, so only a
    shared reason keeps a caller from being told two different things about the
    same pair of rates.
    """
    recorder = _FakeRecorder(fps)
    envelope = dataset_rate_mismatch_error("run_policy", recorder, control_frequency)
    reason = dataset_rate_mismatch_reason("run_policy", recorder, control_frequency)
    assert (envelope is None) == (reason is None), (
        f"the two surfaces disagree for fps={fps!r} control_frequency={control_frequency!r}"
    )
    if envelope is not None and reason is not None:
        assert envelope["content"][0]["text"] == reason


def test_the_reason_names_the_method_the_caller_actually_called():
    """The remedy must advise changing ``PolicyRunner.run``, not ``run_policy``."""
    reason = dataset_rate_mismatch_reason("PolicyRunner.run", _FakeRecorder(30), 50.0)
    assert reason is not None
    assert reason.startswith("PolicyRunner.run: ")
    assert "pass control_frequency=30 to PolicyRunner.run()" in reason
    assert "run_policy" not in reason


class _Dataset:
    def __init__(self, fps, meta: object | None = None) -> None:  # noqa: ANN001
        self.fps = fps
        self.meta = meta


class _Meta:
    def __init__(self, fps) -> None:  # noqa: ANN001
        self.fps = fps


class _FakeRecorder:
    """Exposes the rate the way ``LeRobotDataset`` does, directly."""

    def __init__(self, fps) -> None:  # noqa: ANN001
        self.dataset = _Dataset(fps)


class _MetaOnlyRecorder:
    """A layout that carries the rate only on the metadata object."""

    def __init__(self, fps) -> None:  # noqa: ANN001
        self.dataset = _Dataset(None, meta=_Meta(fps))


_ENTRY_POINTS = {
    "strands_robots/simulation/base.py": ("run_policy", "eval_policy", "evaluate_benchmark"),
    "strands_robots/simulation/mujoco/simulation.py": ("start_policy", "run_multi_policy"),
}

# ``PolicyRunner`` is driven directly too, so it repeats the check under its own
# name. Kept as a separate table because the guard it calls differs: the engine
# returns a tool envelope, the runner raises for a caller that has no envelope.
_RUNNER_ENTRY_POINTS = {
    "strands_robots/simulation/policy_runner.py": ("run", "evaluate"),
}


def _self_calls(module: str, method: str) -> set[str]:
    """Names of the ``self.x(...)`` calls made anywhere inside ``module::method``."""
    tree = ast.parse(Path(module).read_text())
    definitions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method]
    if not definitions:
        pytest.fail(f"{method} not found in {module}")
    return {
        n.func.attr for n in ast.walk(definitions[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }


@pytest.mark.parametrize(
    ("module", "method"),
    [(m, f) for m, fns in _ENTRY_POINTS.items() for f in fns],
)
def test_every_rollout_entry_point_checks_the_recording_rate(module, method):
    """No rollout driver may capture into a dataset it disagrees with.

    Pinned structurally rather than behaviourally so a driver that needs a
    checkpoint, a benchmark registration or a live background thread to reach
    is still covered - and so a newly added driver cannot quietly skip the
    guard the way ``run_multi_policy`` once skipped ``_validate_action_horizon``.
    """
    assert "_validate_recording_rate" in _self_calls(module, method), (
        f"{module}::{method} does not check the recording rate"
    )


@pytest.mark.parametrize(
    ("module", "method"),
    [(m, f) for m, fns in _RUNNER_ENTRY_POINTS.items() for f in fns],
)
def test_every_directly_drivable_runner_method_checks_the_recording_rate(module, method):
    """The runner cannot rely on a guard that is not on a direct caller's path."""
    assert "_reject_recording_rate_mismatch" in _self_calls(module, method), (
        f"{module}::{method} does not check the recording rate"
    )
