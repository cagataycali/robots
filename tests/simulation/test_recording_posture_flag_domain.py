"""A recording posture flag must be a boolean, not anything truthy.

``start_recording`` takes two flags that select a *posture* rather than scaling
a quantity, and both were read by truthiness. Every non-empty string is truthy,
so ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` - the spellings an operator
reaches for when opting out - selected the branch the caller was opting *out*
of, and each failure is unrecoverable:

* ``overwrite="false"`` reached
  :meth:`~strands_robots.simulation.recording.DatasetRecordingMixin._prepare_dataset_target`
  as True and ``shutil.rmtree``-d the dataset. Measured end to end on MuJoCo: a
  dataset holding one recorded episode, re-opened to append a second, came back
  with one episode - the first was deleted - and ``start_recording`` returned
  ``status="success"`` throughout. That method already refuses to clobber a
  non-empty *non*-dataset directory, so the one thing it deleted without asking
  was a real LeRobotDataset.
* ``push_to_hub="false"`` was stashed on the recording state and *published* the
  finished dataset to the Hub at ``stop_recording``. Six values that are not
  booleans uploaded it.

``fps``, in the same signature, has been checked on a shared domain since the
rate defect it caused; these two had none. They are now checked on
:func:`~strands_robots.utils.boolean_flag_error` through the shared
:func:`~strands_robots.simulation.recording.dataset_recording_posture_error`,
ahead of the lerobot-extra probe and ahead of anything on disk, so the same
mistake reports the same way on every install and on every backend.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

import strands_robots.simulation as simulation_pkg
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    dataset_recording_posture_error,
)
from strands_robots.utils import boolean_flag_error

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# One actuated hinge: enough for run_policy to drive something and for the
# dataset schema to declare joint/action columns, with no asset download.
_ARM_XML = """
<mujoco model="posture_flag_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
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

#: Flags whose truthiness used to be read instead of their type.
POSTURE_FLAGS = ["push_to_hub", "overwrite"]

#: Values no posture can be read from. The four string spellings and the
#: non-zero number are the ones that selected the *permissive* branch; ``None``
#: and ``[]`` silently took the other one without ever being a declared
#: spelling of it.
UNUSABLE_FLAGS = [
    "false",
    "no",
    "off",
    "0",
    "true",
    1,
    0,
    "",
    None,
    [],
    float("nan"),
]

#: The truthy non-booleans: each one selected the branch that deletes a dataset
#: (``overwrite``) or publishes it (``push_to_hub``).
TRUTHY_NON_BOOLEANS = ["false", "no", "off", "0", "true", 1, float("nan")]


@pytest.fixture
def sim(tmp_path):
    model = tmp_path / "posture_flag_arm.xml"
    model.write_text(_ARM_XML)
    s = Simulation(tool_name="posture_flag", mesh=False)
    s.create_world()
    s.add_robot("arm", urdf_path=str(model))
    yield s
    s.cleanup()


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


def _episode_count(root: Path) -> int | None:
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    return int(json.loads(info.read_text())["total_episodes"])


def _record_one_episode(sim, root: Path, **kwargs: Any) -> dict[str, Any]:
    """Open a session, drive four control steps, save. Returns the start result."""
    started = sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root), **kwargs)
    if started["status"] == "success":
        rollout = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=4, control_frequency=30.0)
        assert rollout["status"] == "success", rollout
        assert sim.stop_recording()["status"] == "success"
    return started


class TestThePostureDomain:
    """The shared guard is the boolean-flag domain, and nothing more."""

    @pytest.mark.parametrize("param", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", UNUSABLE_FLAGS)
    def test_a_non_boolean_reports_the_flag_the_method_and_the_value(self, param, value):
        error = dataset_recording_posture_error("start_recording", param, value)
        assert error is not None
        assert error["status"] == "error"
        text = error["content"][0]["text"]
        assert "start_recording" in text
        assert param in text
        assert repr(value) in text

    @pytest.mark.parametrize("param", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_accepted(self, param, value):
        assert dataset_recording_posture_error("start_recording", param, value) is None

    @pytest.mark.parametrize("value", [*UNUSABLE_FLAGS, True, False])
    def test_the_guard_adds_nothing_to_the_shared_domain(self, value):
        """The envelope carries the shared message verbatim - no second rule."""
        error = dataset_recording_posture_error("start_recording", "overwrite", value)
        text = boolean_flag_error(value, "overwrite", "start_recording")
        if text is None:
            assert error is None
        else:
            assert error is not None
            assert error["content"][0]["text"] == text


class TestOverwriteNoLongerDeletesTheDatasetItWasOptingOutOf:
    """The measured defect: a truthy non-boolean wiped recorded episodes."""

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_the_recorded_episode_survives_a_refused_overwrite(self, sim, tmp_path, value):
        pytest.importorskip("lerobot")
        root = tmp_path / "dataset"
        assert _record_one_episode(sim, root)["status"] == "success"
        assert _episode_count(root) == 1

        refused = sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root), overwrite=value)

        assert refused["status"] == "error", refused
        assert "overwrite" in _text(refused)
        # The whole point: the caller's episode is still there.
        assert _episode_count(root) == 1, "the refused call deleted the dataset"
        assert (root / "meta").is_dir()

    def test_both_documented_postures_still_do_what_they_say(self, sim, tmp_path):
        """``False`` appends, ``True`` records from scratch."""
        pytest.importorskip("lerobot")
        root = tmp_path / "dataset"
        _record_one_episode(sim, root, overwrite=False)
        _record_one_episode(sim, root, overwrite=False)
        assert _episode_count(root) == 2, "overwrite=False must append"

        _record_one_episode(sim, root, overwrite=True)
        assert _episode_count(root) == 1, "overwrite=True must record from scratch"


class TestPushToHubNoLongerPublishesWhenOptedOut:
    """A publication posture is not read by truthiness."""

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_start_recording_refuses_before_the_session_opens(self, sim, tmp_path, value):
        root = tmp_path / "dataset"
        refused = sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root), push_to_hub=value)

        assert refused["status"] == "error", refused
        assert "push_to_hub" in _text(refused)
        # No half-open session, and nothing on disk: the refusal precedes both.
        assert "idle" in _text(sim.get_recording_status()).lower()
        assert not root.exists() or not any(root.iterdir())

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_stop_recording_refuses_without_uploading(self, sim, tmp_path, monkeypatch, value):
        pytest.importorskip("lerobot")
        import strands_robots.dataset_recorder as dataset_recorder

        uploads: list[Any] = []

        def spy(self, tags=None, private=True):
            uploads.append({"tags": tags, "private": private})
            return {"status": "success", "content": [{"text": "pushed"}]}

        monkeypatch.setattr(dataset_recorder.DatasetRecorder, "push_to_hub", spy)

        root = tmp_path / "dataset"
        started = sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root))
        assert started["status"] == "success", started
        assert (
            sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=4, control_frequency=30.0)["status"]
            == "success"
        )

        refused = sim.stop_recording(push_to_hub=value)

        assert refused["status"] == "error", refused
        assert "push_to_hub" in _text(refused)
        assert uploads == [], "the refused stop_recording published the dataset"

    def test_a_boolean_override_still_publishes(self, sim, tmp_path, monkeypatch):
        pytest.importorskip("lerobot")
        import strands_robots.dataset_recorder as dataset_recorder

        uploads: list[Any] = []
        monkeypatch.setattr(
            dataset_recorder.DatasetRecorder,
            "push_to_hub",
            lambda self, tags=None, private=True: (
                uploads.append(tags) or {"status": "success", "content": [{"text": "pushed"}]}
            ),
        )

        root = tmp_path / "dataset"
        assert sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root))["status"] == "success"
        assert (
            sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=4, control_frequency=30.0)["status"]
            == "success"
        )

        stopped = sim.stop_recording(push_to_hub=True)

        assert stopped["status"] == "success", stopped
        assert len(uploads) == 1

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_the_idle_path_reports_the_flag_too(self, sim, value):
        """No session open: the flag is judged before the idle branch reads it."""
        refused = sim.stop_recording(push_to_hub=value)

        assert refused["status"] == "error", refused
        assert "push_to_hub" in _text(refused)


class TestTheRefusalPrecedesTheLerobotProbe:
    """The same mistake must report the same way on every install."""

    @pytest.mark.parametrize("param", POSTURE_FLAGS)
    def test_the_flag_is_named_even_when_the_dataset_stack_is_missing(self, sim, tmp_path, monkeypatch, param):
        import strands_robots.dataset_recorder as dataset_recorder

        def fatal(*_args, **_kwargs):
            raise AssertionError("the refused call reached the lerobot-extra probe")

        monkeypatch.setattr(dataset_recorder, "lerobot_dataset_import_error", fatal)

        refused = sim.start_recording(
            repo_id="local/posture_flag",
            fps=30,
            root=str(tmp_path / "dataset"),
            **{param: "false"},
        )

        assert refused["status"] == "error"
        assert param in _text(refused)


def _flags_checked_by(source: str, function: str) -> set[str]:
    """Flag names passed to the shared posture guard inside *function*.

    Parsed by AST so backends whose optional dependencies (Isaac Sim, Newton)
    are not installed are still checked.
    """
    checked: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "dataset_recording_posture_error"
            ):
                continue
            for arg in call.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    checked.add(arg.value)
        # A per-flag loop names the flags in the iterated tuple instead.
        for loop in ast.walk(node):
            if not isinstance(loop, ast.For):
                continue
            body = ast.unparse(ast.Module(body=list(loop.body), type_ignores=[]))
            if "dataset_recording_posture_error" not in body:
                continue
            for element in ast.walk(loop.iter):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    checked.add(element.value)
    return checked


def _backend_recording_source(backend: str) -> str:
    path = Path(simulation_pkg.__file__).parent / backend / "recording.py"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", ["mujoco", "newton", "isaac"])
def test_every_backend_start_recording_checks_both_flags(backend):
    """No backend may accept a posture the others refuse."""
    checked = _flags_checked_by(_backend_recording_source(backend), "start_recording")
    assert set(POSTURE_FLAGS) <= checked, (
        f"{backend}/recording.py start_recording must check {POSTURE_FLAGS} "
        f"via dataset_recording_posture_error; found {sorted(checked)}"
    )


def test_the_shared_stop_recording_checks_its_override():
    """The one ``stop_recording`` every backend inherits judges the override."""
    source = Path(inspect.getfile(DatasetRecordingMixin))
    checked = _flags_checked_by(source.read_text(encoding="utf-8"), "stop_recording")
    assert "push_to_hub" in checked, sorted(checked)


def test_the_scanner_detects_an_unchecked_flag():
    """Non-vacuity: a start_recording with no guard must be reported."""
    planted = "def start_recording(self, push_to_hub=False, overwrite=False):\n    return {'status': 'success'}\n"
    assert _flags_checked_by(planted, "start_recording") == set()
