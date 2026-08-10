"""Omitting ``robot_name`` on a motion primitive resolves the sole robot.

``move_to`` / ``set_gripper`` / ``rotate_wrist`` all declare
``robot_name: str | None = None``, and all three document it as "defaults to the
single robot in the world (errors if ambiguous)". The default is implemented
once, in the shared preamble ``_primitive_resolve_robot``, which delegates to
:meth:`strands_robots.simulation.base.SimEngine._resolve_single_robot` - whose
three documented outcomes are one robot -> that robot, zero robots ->
``ValueError``, many -> ``ValueError`` listing the candidates so the caller can
recover in zero extra calls - and converts either ``ValueError`` into the
tool-error envelope, because each primitive also documents "Never raises.".

Every other primitive test passes ``robot_name`` explicitly, so none of that had
coverage: not the default itself, not the ambiguity refusal, and not the envelope
conversion standing between a bare ``ValueError`` and three methods documented
never to raise. A silently-resolved wrong robot is the failure mode that matters
here - with two arms in the scene, resolving either one without being asked
drives a robot the caller never named - so the ambiguity refusal is pinned as
carefully as the success.

All three outcomes are pinned on each primitive against a real MuJoCo scene (the
inline MJCF arm shared with ``test_motion_primitives`` - no asset downloads).
Resolution runs before any IK solve, so only the tests that expect a ``move_to``
success or refusal-about-the-target ``importorskip`` on ``mink``; the resolution
refusals themselves run on bare ``mujoco``.
"""

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

PRIMITIVES = ("move_to", "set_gripper", "rotate_wrist")

# The smallest converging call per primitive. ``tol`` is looser than the residual
# this 4-DOF arm leaves on ``REACHABLE`` so ``move_to`` enters its servo loop
# instead of refusing the target up front.
_CALLS: dict[str, dict[str, Any]] = {
    "move_to": {"position": REACHABLE, "tol": 0.05, "max_steps": 400},
    "set_gripper": {"state": "open", "steps": 5},
    "rotate_wrist": {"target_yaw": 0.3, "tol": 0.05, "max_steps": 300},
}


def _text(result: dict[str, Any]) -> str:
    """Flatten the envelope's text blocks - the report a caller actually reads."""
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


def _invoke(sim: Any, primitive: str, **extra: Any) -> dict[str, Any]:
    """Call ``primitive`` with its minimal arguments; ``extra`` adds ``robot_name``."""
    return getattr(sim, primitive)(**_CALLS[primitive], **extra)


def _skip_without_ik(primitive: str) -> None:
    """``move_to`` needs the mink bridge to judge a target; resolution does not."""
    if primitive == "move_to":
        pytest.importorskip("mink")


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def scene(arm_path):
    """Factory building a world with ``n`` copies of the arm, spaced 0.8 m apart."""
    built: list[Simulation] = []

    def make(n_robots: int) -> Simulation:
        s = Simulation(tool_name="test_primitive_robot_auto_resolution", mesh=False)
        assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
        for i in range(n_robots):
            name = "arm" if i == 0 else f"arm{i + 1}"
            placed = s.add_robot(name, urdf_path=arm_path, position=[0.8 * i, 0.0, 0.0])
            assert placed["status"] == "success", placed
        built.append(s)
        return s

    yield make
    for s in built:
        s.cleanup(policy_stop_timeout=2.0)


class TestTheSoleRobotIsResolved:
    """One robot and no ``robot_name`` drives that robot, per all three docstrings."""

    @pytest.mark.parametrize("primitive", PRIMITIVES)
    def test_omitting_robot_name_drives_the_sole_robot(self, scene, primitive):
        _skip_without_ik(primitive)
        result = _invoke(scene(1), primitive)
        assert result["status"] == "success", _text(result)
        assert "'arm'" in _text(result)

    @pytest.mark.parametrize("primitive", PRIMITIVES)
    def test_omitting_is_the_same_call_as_naming_the_sole_robot(self, scene, primitive):
        """The default is a resolution, not a second code path.

        Two identical fresh worlds are deterministic, so the whole envelope -
        text and json - is compared rather than just the status.
        """
        _skip_without_ik(primitive)
        omitted = _invoke(scene(1), primitive)
        named = _invoke(scene(1), primitive, robot_name="arm")
        assert omitted["status"] == "success", _text(omitted)
        assert omitted == named


class TestAnEmptyWorldIsRefusedWithTheRecovery:
    """Zero robots and no ``robot_name``: an envelope naming the fix, not a raise."""

    @pytest.mark.parametrize("primitive", PRIMITIVES)
    def test_an_empty_world_reports_how_to_recover(self, scene, primitive):
        result = _invoke(scene(0), primitive)
        assert result["status"] == "error"
        text = _text(result)
        assert "No robots registered" in text
        assert "add_robot" in text


class TestAmbiguityIsRefusedRatherThanGuessed:
    """Many robots and no ``robot_name``: refuse, and list the candidates.

    Resolving either arm would drive one the caller never named, so the refusal -
    and the candidate list that lets the caller correct it in zero extra calls -
    is the contract rather than the convenience.
    """

    @pytest.mark.parametrize("primitive", PRIMITIVES)
    def test_an_ambiguous_scene_names_the_parameter_and_the_candidates(self, scene, primitive):
        result = _invoke(scene(2), primitive)
        assert result["status"] == "error"
        text = _text(result)
        assert "robot_name" in text
        assert "'arm'" in text
        assert "'arm2'" in text

    @pytest.mark.parametrize("primitive", PRIMITIVES)
    def test_naming_a_robot_still_routes_to_it_when_the_scene_is_ambiguous(self, scene, primitive):
        """The over-reach control: only an OMITTED name is ambiguous.

        Asserted on which robot the report names rather than on the status: the
        second arm stands 0.8 m from the shared ``REACHABLE`` target, so
        ``move_to`` correctly reports that target unreachable for it - and naming
        ``arm2`` in that refusal is itself the proof the call routed there.
        """
        _skip_without_ik(primitive)
        result = _invoke(scene(2), primitive, robot_name="arm2")
        text = _text(result)
        assert "'arm2'" in text
        assert "Multiple robots registered" not in text
