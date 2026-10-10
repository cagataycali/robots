"""Every backend's ``start_recording`` must *return* the refusals it wires.

``start_recording`` makes four caller-input refusals before it touches a
dataset - the ``fps`` domain, the two posture flags, the ``cameras`` name list
and a rate an in-flight rollout is not capturing at - and each one is a domain
shared with the other backends. Three sibling contract modules already prove
every backend *calls* the shared guard:

* ``test_dataset_recording_fps_contract.py`` for ``fps``,
* ``test_recording_posture_flag_domain.py`` for ``push_to_hub`` / ``overwrite``,
* ``test_camera_name_list_contract.py`` for ``cameras``,
* ``test_recording_rate_matches_control_frequency.py`` for the rollout rate.

All four do it by parsing the module with ``ast``, on the stated grounds that
"the Isaac and Newton backends need their simulators installed to drive". That
reason is not what the tree shows: ``tests/simulation/isaac/test_dataset_recording.py``
and ``tests/simulation/newton/test_dataset_recording.py`` both drive the same
method end to end through a skeleton engine built with ``__new__``, and each of
those modules says so in its own docstring - "without the Isaac Sim Kit
runtime", "without the optional Newton/Warp physics stack". The three guards
that reject caller input also sit *above* the lerobot-extra probe by design, so
none of them needs the dataset stack either.

A structural pin proves the guard is *called*, never that its refusal is
*returned*: keeping the call and discarding the ``return`` satisfies it. On the
one backend a driver had ever exercised (MuJoCo) all four refusals were
returned; on the two the sweeps alone covered, none of the four had ever run.
This module drives them, closing the six reachable cells:

===============  ===========  ===========  ===========
refusal          MuJoCo       Newton       Isaac
===============  ===========  ===========  ===========
``fps``          driven       **added**    **added**
posture flags    driven       **added**    **added**
``cameras``      driven       **added**    **added**
rollout rate     driven       unreachable  unreachable
===============  ===========  ===========  ===========

The fourth cell is *provably* unreachable on these two backends rather than
untested: the guard compares against
:meth:`~strands_robots.simulation.base.SimEngine._active_rollout_rates`, which
only the MuJoCo backend overrides, so both of these inherit an empty mapping
and the guard returns ``None`` for every ``fps``. Its structural pin is the
right tool for it and stays - it is what stops a backend that grows an
asynchronous rollout later from inheriting a ``start_recording`` that skipped
the check.

Runs on a minimal install: the refusals precede the lerobot-extra probe, and
none of the imports here pulls MuJoCo, Isaac Sim, Newton, Warp or lerobot in
(pinned below).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import _RobotState
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.simulation.recording import (
    dataset_recording_option_error,
    dataset_recording_posture_error,
)
from strands_robots.utils import name_list_error
from tests.simulation._isaac_engine import isaac_engine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Probe values are declared here rather than imported from the three contract
# modules: each of those does ``pytest.importorskip("mujoco")`` at module level,
# and importing a constant from one would inherit that skip - on exactly the
# install these refusals have to hold on. TestTheProbeValuesAreOutsideTheSharedDomains
# below pins each list against the shared domain that owns it, so a value that
# stops being refused fails here instead of silently weakening a case.
UNUSABLE_FPS: list[Any] = [0, -5, 2.7, float("nan"), float("inf"), True, "30", None, [30]]
POSTURE_FLAGS = ("push_to_hub", "overwrite")
TRUTHY_NON_BOOLEANS: list[Any] = ["false", "no", "off", "0", 1, float("nan")]
UNUSABLE_CAMERA_LISTS: list[Any] = ["wrist", ["front", "front"], {"front": 1}, ["front", 3], ["front", ""]]
NON_STRING_TASKS: list[Any] = [None, 0, False, 123, 3.14, ["pick", "cube"], {"a": 1}, b"pick"]


def _isaac_engine() -> Any:
    """A skeleton ``IsaacSimulation`` with no Isaac Sim Kit runtime.

    Same shape as the harness ``tests/simulation/isaac/test_dataset_recording.py``
    uses: ``__new__`` plus exactly the attributes the recording path reads, so
    the fixture stays honest about what ``start_recording`` depends on.
    """
    engine = isaac_engine()
    engine._config = IsaacConfig(render_mode="rtx_realtime")
    engine._world_created = True
    engine._robots = {
        "so100": _RobotState(
            name="so100",
            prim_path="/World/Robots/so100",
            joint_names=list(_SO100_JOINTS),
            data_config="so100",
        )
    }
    return engine


def _newton_engine() -> Any:
    """A ``NewtonSimEngine`` over a hand-built world, with no Newton/Warp stack."""
    world = SimWorld()
    world.robots["so100"] = SimRobot(
        name="so100", urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
    )
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


BACKENDS = [
    pytest.param(_isaac_engine, id="isaac"),
    pytest.param(_newton_engine, id="newton"),
]


def _start(factory: Any, root: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """Call ``start_recording`` on a fresh engine from *factory*.

    ``**kwargs`` carries deliberately unusable values, so it is typed ``Any``
    rather than restated per parameter.
    """
    target = str(root) if root is not None else "/tmp/strands-never-created"
    return factory().start_recording(repo_id="local/preflight_probe", root=target, **kwargs)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


class TestTheProbeValuesAreOutsideTheSharedDomains:
    """Non-vacuity: every probe value really is one its shared domain refuses."""

    def test_every_probe_is_refused_and_a_usable_value_accepted(self) -> None:
        for fps in UNUSABLE_FPS:
            assert dataset_recording_option_error("start_recording", fps) is not None, repr(fps)
        for flag in POSTURE_FLAGS:
            for value in TRUTHY_NON_BOOLEANS:
                assert dataset_recording_posture_error("start_recording", flag, value) is not None, (flag, value)
        for cameras in UNUSABLE_CAMERA_LISTS:
            assert name_list_error(cameras, "cameras", "start_recording") is not None, repr(cameras)
        # Over-reach control: the domains do not refuse everything.
        assert dataset_recording_option_error("start_recording", 30) is None
        assert dataset_recording_posture_error("start_recording", "overwrite", True) is None
        assert name_list_error(["front", "wrist"], "cameras", "start_recording") is None


@pytest.mark.parametrize("factory", BACKENDS)
class TestEveryBackendReturnsTheSharedRefusal:
    """Each refusal is returned, names its parameter, and is the shared verdict verbatim.

    One cell per backend and refusal: the probe values loop inside it, since
    each is a fact about the shared domain (pinned above), not about the backend.
    """

    def test_an_unusable_fps_is_refused(self, factory: Any) -> None:
        """A rate no dataset can be written at is refused, not reported as started."""
        for fps in UNUSABLE_FPS:
            result = _start(factory, fps=fps)
            assert result["status"] == "error", repr(fps)
            assert "fps" in _text(result), repr(fps)
            assert result == dataset_recording_option_error("start_recording", fps), repr(fps)

    def test_a_truthy_posture_opt_out_is_refused(self, factory: Any) -> None:
        """A posture flag is checked, not parsed - a truthy opt-out is refused."""
        for flag in POSTURE_FLAGS:
            for value in TRUTHY_NON_BOOLEANS:
                posture: dict[str, Any] = {flag: value}
                result = _start(factory, **posture)
                assert result["status"] == "error", (flag, value)
                assert flag in _text(result), (flag, value)
                assert result == dataset_recording_posture_error("start_recording", flag, value), (flag, value)

    def test_an_unusable_camera_list_is_refused(self, factory: Any) -> None:
        """``cameras`` is an ordered list of distinct names on every backend."""
        for cameras in UNUSABLE_CAMERA_LISTS:
            result = _start(factory, cameras=cameras)
            assert result["status"] == "error", repr(cameras)
            assert _text(result) == name_list_error(cameras, "cameras", "start_recording"), repr(cameras)

    def test_a_non_string_task_is_refused(self, factory: Any) -> None:
        """``task`` labels the recorded ``task`` column, so it is a string - as ``instruction`` is."""
        for task in NON_STRING_TASKS:
            result = _start(factory, task=task)
            assert result["status"] == "error", repr(task)
            assert _text(result) == f"start_recording: 'task' must be a string, got {type(task).__name__}."

    def test_a_refused_start_touches_no_dataset(
        self, factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each refusal is returned before anything on disk or in state moves.

        Placement, in the guards' own words: "before the lerobot-extra probe".
        The probe is made fatal with an ``AssertionError`` - the enclosing
        ``except ImportError`` cannot swallow it - so reaching it fails loudly
        instead of degrading into the extra-missing report.
        """
        from strands_robots import dataset_recorder

        def _fatal() -> str | None:
            raise AssertionError("the refusal must precede the lerobot-extra probe")

        monkeypatch.setattr(dataset_recorder, "lerobot_dataset_import_error", _fatal)
        refusals: list[dict[str, Any]] = [
            {"fps": 2.7},
            {"overwrite": "false"},
            {"push_to_hub": "no"},
            {"cameras": "wrist"},
            {"task": None},
        ]
        for kwargs in refusals:
            engine = factory()
            target = tmp_path / "never"
            assert engine.start_recording(repo_id="local/p", root=str(target), **kwargs)["status"] == "error", kwargs
            assert not target.exists(), kwargs
            state = engine._recording_state()
            assert state is not None
            assert not state.get("recording"), kwargs

    def test_the_rolling_rate_refusal_is_unreachable(self, factory: Any) -> None:
        """The fourth refusal is provably dead here, so its structural pin is right."""
        engine = factory()
        assert engine._active_rollout_rates() == {}
        for fps in (30, 50, 1):
            assert engine._validate_recording_start_rate(fps, "start_recording") is None


def test_mujoco_is_the_backend_that_overrides_the_rate_source() -> None:
    """Non-vacuity: the empty rollout-rate mapping above is inherited, not universal.

    Read from the source so the assertion holds without a MuJoCo install.
    """
    import strands_robots.simulation as simulation_pkg

    root = Path(simulation_pkg.__file__).parent
    overriding = {
        backend
        for backend in ("mujoco", "newton", "isaac")
        for module in sorted((root / backend).glob("*.py"))
        if "def _active_rollout_rates(" in module.read_text(encoding="utf-8")
    }
    assert overriding == {"mujoco"}, overriding


class TestTheRefusalsNeedNoOptionalDependency:
    """The whole module runs on an install with none of the simulators."""

    def test_importing_the_surfaces_pulls_no_heavy_module(self) -> None:
        """Measured in a child interpreter: this module's imports stay light."""
        program = (
            "import sys\n"
            "from strands_robots.simulation.isaac.simulation import IsaacSimulation\n"
            "from strands_robots.simulation.newton.simulation import NewtonSimEngine\n"
            "from strands_robots.simulation.recording import dataset_recording_option_error\n"
            "from strands_robots.utils import name_list_error\n"
            "heavy = ('mujoco', 'lerobot', 'newton', 'warp', 'isaacsim', 'omni')\n"
            "print(sorted(m for m in heavy if m in sys.modules))\n"
        )
        out = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True, timeout=180)
        assert out.stdout.strip().endswith("[]"), out.stdout
