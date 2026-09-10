"""``target_velocity`` is bounded by one envelope at the wire AND at the sink (F-005, CWE-20).

The mesh ``start`` / ``execute`` validator coerced each component into the
``+/-1e6`` domain it shares with ``target_pose`` - a coordinate range, not a
speed - and the WBC policy's ``_validate_velocity`` checked only finiteness
before multiplying the value by ``cmd_scale`` straight into the observation. A
``[1e6, 0, 0]`` was a valid command at both layers. Sibling controls in the
same validator (``control_frequency``, ``steps``) are tightly bounded.

Both readers now import :mod:`strands_robots.locomotion_envelope`, so the wire
and the sink cannot disagree, a caller that bypasses the mesh meets the same
refusal, and the bound is refused - never clamped - with a reason naming the
component, value, bound, unit and the environment variable that raises it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from strands_robots import locomotion_envelope as env
from strands_robots.mesh import security as sec
from strands_robots.policies.wbc.policy import WBCPolicy

ROOT = Path(__file__).resolve().parents[1]


def _base() -> dict[str, object]:
    return {"action": "execute", "instruction": "walk", "policy_provider": "mock"}


@pytest.fixture(autouse=True)
def _default_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LINEAR_ENV_VAR, raising=False)
    monkeypatch.delenv(env.ANGULAR_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# The envelope module itself.
# ---------------------------------------------------------------------------


def test_the_envelope_module_is_standard_library_only() -> None:
    """Both readers sit in layers that must not import each other; the shared module stays light."""
    tree = ast.parse((ROOT / "strands_robots" / "locomotion_envelope.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"math", "os", "__future__"}, imported


def test_component_two_is_angular_and_the_rest_linear() -> None:
    assert env.component_bound(0) == (2.0, "m/s")
    assert env.component_bound(1) == (2.0, "m/s")
    assert env.component_bound(2) == (2.0, "rad/s")
    assert env.component_bound(3) == (2.0, "m/s"), "any component past the third takes the linear bound"


def test_the_bound_is_inclusive_and_signed() -> None:
    assert env.target_velocity_component_error(0, 2.0, "t") is None
    assert env.target_velocity_component_error(0, -2.0, "t") is None
    assert env.target_velocity_component_error(0, 2.0000001, "t") is not None
    assert env.target_velocity_component_error(0, -2.0000001, "t") is not None


def test_the_refusal_names_component_value_bound_unit_and_the_knob() -> None:
    reason = env.target_velocity_component_error(2, 5.0, "validate_command")
    assert reason is not None
    for fragment in ("validate_command", "target_velocity[2]", "5.0", "2.0", "rad/s", env.ANGULAR_ENV_VAR, "refusing"):
        assert fragment in reason, fragment


@pytest.mark.parametrize("raw", ["", "abc", "-1", "0", "nan", "inf"])
def test_an_unusable_env_override_leaves_the_default_in_force(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(env.LINEAR_ENV_VAR, raw)
    assert env.max_linear_velocity_mps() == env.MAX_TARGET_LINEAR_VELOCITY_MPS


def test_an_operator_can_raise_the_bound_without_a_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env.target_velocity_component_error(0, 3.0, "t") is not None
    monkeypatch.setenv(env.LINEAR_ENV_VAR, "5")
    assert env.target_velocity_component_error(0, 3.0, "t") is None
    assert env.target_velocity_component_error(2, 3.0, "t") is not None, "the angular bound is its own knob"


# ---------------------------------------------------------------------------
# The wire: mesh validate_command.
# ---------------------------------------------------------------------------


class TestTheMeshValidator:
    @pytest.mark.parametrize(
        "velocity",
        [
            pytest.param([1e6, 0.0, 0.0], id="the-old-pose-domain-ceiling"),
            pytest.param([0.0, -2.5, 0.0], id="lateral-past-linear"),
            pytest.param([0.0, 0.0, 2.1], id="omega-past-angular"),
            pytest.param([0.5, 0.0, 0.0, 3.0], id="fourth-component-past-linear"),
        ],
    )
    def test_an_out_of_envelope_component_is_refused_not_clamped(self, velocity: list[float]) -> None:
        with pytest.raises(sec.ValidationError, match="locomotion envelope") as info:
            sec.validate_command({**_base(), "target_velocity": velocity})
        assert "target_velocity[" in str(info.value)
        assert "refusing" in str(info.value)

    def test_the_bounds_pass_exactly(self) -> None:
        out = sec.validate_command({**_base(), "target_velocity": [2.0, -2.0, 2.0]})
        assert out["target_velocity"] == [2.0, -2.0, 2.0]

    def test_a_walking_command_still_passes(self) -> None:
        out = sec.validate_command({**_base(), "target_velocity": [0.5, 0.0, 0.2]})
        assert out["target_velocity"] == [0.5, 0.0, 0.2]

    def test_a_non_finite_component_is_still_refused_before_the_envelope(self) -> None:
        with pytest.raises(sec.ValidationError, match="target_velocity"):
            sec.validate_command({**_base(), "target_velocity": [float("nan"), 0.0, 0.0]})

    def test_the_operator_knob_is_honoured_on_the_wire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(env.LINEAR_ENV_VAR, "4")
        out = sec.validate_command({**_base(), "target_velocity": [3.5, 0.0, 0.0]})
        assert out["target_velocity"] == [3.5, 0.0, 0.0]


# ---------------------------------------------------------------------------
# The sink: WBCPolicy._validate_velocity.
# ---------------------------------------------------------------------------


class TestTheWbcSink:
    @pytest.mark.parametrize(
        "velocity",
        [
            pytest.param([1e6, 0.0, 0.0], id="the-old-pose-domain-ceiling"),
            pytest.param([0.0, 2.5, 0.0], id="lateral-past-linear"),
            pytest.param([0.0, 0.0, -2.1], id="omega-past-angular"),
        ],
    )
    def test_the_policy_refuses_the_same_values_the_wire_refuses(self, velocity: list[float]) -> None:
        with pytest.raises(ValueError, match="locomotion envelope") as info:
            WBCPolicy._validate_velocity(velocity)
        assert "WBCPolicy" in str(info.value)

    def test_the_policy_accepts_the_bounds_and_a_walk(self) -> None:
        assert list(WBCPolicy._validate_velocity([2.0, -2.0, 2.0])) == [2.0, -2.0, 2.0]
        assert list(WBCPolicy._validate_velocity([0.5, 0.0, 0.2])) == [0.5, 0.0, 0.2]

    def test_the_policy_still_refuses_a_non_finite_component_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"target_velocity\[0\].*finite"):
            WBCPolicy._validate_velocity([float("inf"), 0.0, 0.0])

    def test_the_operator_knob_is_honoured_at_the_sink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(env.ANGULAR_ENV_VAR, "3")
        assert list(WBCPolicy._validate_velocity([0.0, 0.0, 2.5])) == [0.0, 0.0, 2.5]


def test_wire_and_sink_agree_on_every_probe() -> None:
    """The single-source promise, graded: no value passes one layer and fails the other."""
    probes = [
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [-2.0, -2.0, -2.0],
        [2.01, 0.0, 0.0],
        [0.0, 0.0, 2.01],
        [1.9, 1.9, 1.9],
        [1e6, 0.0, 0.0],
    ]
    for velocity in probes:
        wire_ok = True
        try:
            sec.validate_command({**_base(), "target_velocity": velocity})
        except sec.ValidationError:
            wire_ok = False
        sink_ok = True
        try:
            WBCPolicy._validate_velocity(velocity)
        except ValueError:
            sink_ok = False
        assert wire_ok == sink_ok, velocity
