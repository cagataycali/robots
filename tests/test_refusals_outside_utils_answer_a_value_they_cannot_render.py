"""Every guard in the package answers a value it cannot render, not only ``utils``'.

``strands_robots.utils`` established the rule: a guard's whole purpose is to
report a caller's bad value through a returned ``{status, content}`` result or a
refusal string, so building that answer must not raise. Rendering a value can
raise - ``repr`` of an ``int`` wider than :func:`sys.get_int_max_str_digits`
raises ``ValueError``, and :class:`numbers.Real` is a registration rather than an
inheritance, so a scalar that satisfies a guard's type test owes it nothing else.
:func:`~strands_robots.utils.refusal_repr` and its two siblings exist for that,
and ``tests/test_refusal_messages_never_raise.py`` pins them.

That scan read one file, and the same guard shape - a value annotated ``Any``
beside the ``str`` labels its call site supplies - is written in nineteen other
modules. Measured on ``4e73dc0f``, forty-three functions across twenty modules
interpolated the caller's value straight into their text, and **twenty-one of them
raised** when called with a value they had already decided to refuse:

| module | guards | raised |
| --- | --- | --- |
| ``simulation/base.py`` | 8 | 5 |
| ``simulation/mujoco/physics.py`` | 3 | 3 |
| ``rendering/compositor.py`` | 5 | 5 |
| ``rendering/video.py`` | 4 | 3 |
| ``simulation/motion_primitives_base.py`` | 3 | 0 |
| ``tools/serial_tool.py`` | 3 | 2 |
| ... 14 more | 17 | 3 |

The other twenty-two offend the rule without this probe reaching them: they lead
with a ``utils`` guard, so a value refused on its *type* is answered by the shared
renderer before their own range branch renders anything. Their own branch is
reached by a value that passes the delegate and fails the range - which is
:class:`TestAStdlibFractionReachesTheRangeBranch`, and needs no third-party type
at all.

The structural half of this - that no guard in the package renders a caller value
directly - is stated over the package in
``tests/test_refusal_messages_never_raise.py``. What is pinned here is the
behaviour that rule is for: each module answers.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from fractions import Fraction
from typing import Any, NamedTuple, cast

import pytest

from strands_robots import hardware_robot, ros_telemetry, streaming_dataset
from strands_robots.drivers import base as drivers_base
from strands_robots.policies.kimodo import config as kimodo_config
from strands_robots.policies.wbc import config as wbc_config
from strands_robots.rendering import compositor, video
from strands_robots.simulation import base as sim_base
from strands_robots.simulation import ik, motion_primitives_base, policy_runner
from strands_robots.simulation.mujoco import physics, scene_ops
from strands_robots.training import _validate

# ``strands_robots.tools.__init__`` binds each tool module's name to the
# ``@tool``-decorated callable it exports, so ``from ... import serial_tool``
# yields the tool rather than the module these guards live in.
lerobot_camera = importlib.import_module("strands_robots.tools.lerobot_camera")
pose_tool = importlib.import_module("strands_robots.tools.pose_tool")
serial_tool = importlib.import_module("strands_robots.tools.serial_tool")


class UnrenderableText(str):
    """A ``str`` subclass whose ``repr`` raises.

    The probe for a guard whose render sits behind a ``str`` test - refusing a
    string *for being* a string, because it would be consumed one character per
    entry. A ``str`` subclass passes that test, so "a ``str``'s ``repr`` cannot
    raise" is true of ``str`` and not of the branch guarding it. It also carries a
    value, so it satisfies a membership test against a set of refused spellings.
    """

    def __repr__(self) -> str:
        raise RuntimeError("this text cannot render itself")


class _Driver:
    """The minimum a verb refusal reads: a tool spec declaring one verb."""

    tool_spec = {"name": "fake_driver", "inputSchema": {"json": {"properties": {"action": {"enum": ["wave"]}}}}}


class Unprintable:
    """A plain object whose ``repr`` raises.

    Refused by every guard below on type alone, so it reaches each one's message
    without depending on that guard's domain. The reason the guarantee is
    unconditional rather than a list of the exceptions known today: a third-party
    type may raise anything at all from its own ``__repr__``.
    """

    def __repr__(self) -> str:
        raise RuntimeError("this type cannot render itself")


#: A ``Fraction`` that does not reduce, because consecutive integers are coprime.
#: ``float()`` of it is ``-1.0`` - finite, inside the float64 range and a
#: ``numbers.Real`` - so it passes the ``utils`` guard a range check leads with,
#: and reaches the range branch. Its ``repr`` renders two 5001-digit integers,
#: which is past :func:`sys.get_int_max_str_digits`, so it raises ``ValueError``.
#: Stdlib only: no registered type and no hostile ``__repr__``.
UNRENDERABLE_FRACTION = Fraction(-(10**5000), 10**5000 + 1)


class Guard(NamedTuple):
    """One guard outside ``utils``, with everything needed to probe it.

    Attributes:
        module: Its module path within the package, which is the test ID.
        call: The guard with its label arguments already bound, so a probe can be
            sent to the position that carries the caller's value.
        probe: The value to send. Usually :class:`Unprintable`, refused on type by
            every guard that leads with one; :class:`UnrenderableText` for the
            three whose render sits behind a ``str`` test, which turns that one
            away before it reaches any text.
        shows: What the answer must say about the value it could not render. A
            container guard renders elementwise, so a refused string is reported as
            its characters - which is the refusal's own reason - rather than as one
            opaque description.
    """

    module: str
    call: Callable[[Any], Any]
    probe: Any
    shows: str


def _text(answer: Any) -> str:
    """The refusal text, whether the guard returns a string or an envelope.

    These guards answer in three shapes - a refusal string the caller wraps, a
    ``{status, content}`` envelope already wrapped, or a ``(value, error)`` pair
    from a coercion - and the property pinned is the same for all three, so it is
    read out of any of them rather than split into three tables.
    """
    assert answer is not None, "an unrenderable value is not a usable one"
    if isinstance(answer, tuple):
        assert answer[0] in (None, 0), f"an unrenderable value was coerced to {answer[0]!r}"
        answer = answer[1]
        assert answer is not None, "a coercion refused without saying why"
    if isinstance(answer, str):
        return answer
    assert answer["status"] == "error", answer
    return " ".join(block["text"] for block in answer["content"])


OPAQUE = "<unrepresentable Unprintable>"

GUARDS: tuple[Guard, ...] = (
    Guard(
        "drivers/base.py",
        lambda v: drivers_base.undeclared_verb_error(_Driver(), v),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "hardware_robot.py",
        lambda v: hardware_robot.Robot._policy_port_error(v, "start_task", "mock"),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "policies/kimodo/config.py",
        lambda v: kimodo_config.sampling_seed_error(v, "KimodoConfig"),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "policies/wbc/config.py",
        lambda v: wbc_config._non_negative_number_error(v, "gain", "WbcConfig"),
        Unprintable(),
        OPAQUE,
    ),
    Guard("rendering/compositor.py", lambda v: compositor._depth_epsilon_error(v), Unprintable(), OPAQUE),
    Guard("rendering/video.py", lambda v: video._stream_rate_error(v), Unprintable(), OPAQUE),
    Guard(
        "ros_telemetry.py",
        lambda v: ros_telemetry._qos_history_depth_error(v, "depth", "QoS"),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "simulation/base.py",
        lambda v: sim_base.finite_non_negative_error(v, "mass", "set_mass"),
        Unprintable(),
        OPAQUE,
    ),
    Guard("simulation/ik.py", lambda v: ik._damping_error(v, "solve_ik"), Unprintable(), OPAQUE),
    Guard(
        "simulation/motion_primitives_base.py",
        # Called unbound: this guard refuses ``state`` before it reads ``self``,
        # so probing it needs no live simulation - and a probe that needed one
        # would be measuring the backend rather than the refusal.
        lambda v: cast(Any, motion_primitives_base.MotionPrimitivesCore._validate_set_gripper_args)(None, v, 1),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "simulation/mujoco/physics.py",
        lambda v: physics._coerce_excluded_body(v, "raycast", 8),
        Unprintable(),
        OPAQUE,
    ),
    Guard(
        "simulation/mujoco/scene_ops.py",
        lambda v: scene_ops._geom_shape_error(v),
        UnrenderableText("mesh"),
        "<unrepresentable UnrenderableText>",
    ),
    Guard(
        "simulation/policy_runner.py",
        lambda v: policy_runner._validate_action_key_map(v),
        UnrenderableText("ab"),
        "['a', 'b']",
    ),
    Guard("streaming_dataset.py", lambda v: streaming_dataset._tolerance_error(v), Unprintable(), OPAQUE),
    Guard(
        "tools/lerobot_camera.py",
        lambda v: lerobot_camera._camera_ids_error(v),
        UnrenderableText("ab"),
        "['a', 'b']",
    ),
    Guard(
        "tools/pose_tool.py",
        lambda v: pose_tool._joint_target_error("wave", "target", "elbow", v),
        Unprintable(),
        OPAQUE,
    ),
    Guard("tools/serial_tool.py", lambda v: serial_tool._motor_id_error(v, "motor_id", "read"), Unprintable(), OPAQUE),
    Guard(
        "training/_validate.py",
        lambda v: _validate._closed_unit_interval_error(v, "gamma", "train"),
        Unprintable(),
        OPAQUE,
    ),
)

GUARD_IDS = tuple(guard.module for guard in GUARDS)


class TestEveryModuleAnswersAValueItCannotRender:
    """One guard per rewritten module, called with a value it refuses on type.

    Every row raises the probe's own ``RuntimeError`` on pre-fix code, in place of
    the answer the guard had already decided to return.
    """

    @pytest.mark.parametrize("guard", GUARDS, ids=GUARD_IDS)
    def test_an_unrenderable_value_is_answered(self, guard: Guard) -> None:
        text = _text(guard.call(guard.probe))
        assert guard.shows in text, "the answer must say what it could not render"
        text.encode("ascii")

    def test_every_module_that_was_rewritten_has_a_row(self) -> None:
        """The table is the module list, so a module cannot be dropped silently."""
        from tests.test_refusal_messages_never_raise import REWRITTEN_MODULES

        assert {guard.module for guard in GUARDS} == REWRITTEN_MODULES - {"utils.py"}

    def test_the_probes_still_refuse_to_render_themselves(self) -> None:
        """Non-vacuity: a probe that had started rendering would pass every row."""
        with pytest.raises(RuntimeError, match="cannot render itself"):
            repr(Unprintable())
        with pytest.raises(RuntimeError, match="cannot render itself"):
            repr(UnrenderableText("mesh"))
        assert UnrenderableText("mesh") == "mesh", "the text probe must still carry its value"


class TestAStdlibFractionReachesTheRangeBranch:
    """The other half of the family: a guard that delegates, then checks a range.

    These lead with a ``utils`` guard, so a value refused on its *type* never
    reaches their own text and the probe above passes them unchanged. Their own
    branch is reached by a value the delegate **accepts** - finite, real, inside the
    float64 range - and the range then refuses. ``Fraction`` is stdlib, so this
    needs no registered type and no hostile ``__repr__``: the escape is reachable
    with two integer literals.
    """

    def test_a_fraction_outside_the_closed_unit_is_answered(self) -> None:
        answer = _validate._closed_unit_interval_error(UNRENDERABLE_FRACTION, "gamma", "train")
        assert answer == "train: gamma must be in [0, 1], got <unrepresentable Fraction>."

    def test_a_fraction_outside_the_half_open_unit_is_answered(self) -> None:
        answer = _validate._half_open_unit_interval_error(UNRENDERABLE_FRACTION, "tau", "train")
        assert answer == "train: tau must be in (0, 1], got <unrepresentable Fraction>."

    def test_the_delegate_accepts_the_probe_so_the_range_branch_is_what_answers(self) -> None:
        """The reason this probe reaches a branch ``Unprintable`` cannot.

        Without this the two rows above would also pass if the delegate had started
        refusing it - a different branch answering, and the range branch still
        unreached.
        """
        from strands_robots.utils import finite_number_error

        assert finite_number_error(UNRENDERABLE_FRACTION, "gamma", "train") is None
        assert float(UNRENDERABLE_FRACTION) == -1.0

    def test_the_probe_is_stdlib_and_cannot_render_itself(self) -> None:
        """Non-vacuity, and the claim that no third-party type is needed."""
        assert type(UNRENDERABLE_FRACTION) is Fraction
        with pytest.raises(ValueError, match="Exceeds the limit"):
            repr(UNRENDERABLE_FRACTION)


class TestTheComponentsThatRenderAreNotErasedWithTheOneThatCannot:
    """A container guard needs a rendering, not a whole-value fallback.

    ``repr`` of a list recurses into its elements, so a container is unrenderable
    whenever any *one* element is, and answering that with
    ``<unrepresentable list>`` erases every element that printed fine - along with
    the element count, which is frequently the refusal's entire reason. The
    container guards outside ``utils`` route through ``refusal_container_repr`` for
    that reason, and this is the property that distinguishes it from the scalar
    renderer.
    """

    def test_a_randomization_range_keeps_the_bound_that_renders(self) -> None:
        answer = sim_base.randomization_range_error([0.5, Unprintable()], "mass_range")
        assert answer is not None
        assert "[0.5, <unrepresentable Unprintable>]" in answer

    def test_a_ray_batch_keeps_the_directions_that_render(self) -> None:
        _, error = physics._coerce_finite_vector([1.0, Unprintable(), 3.0], "origin", "raycast")
        assert "[1.0, <unrepresentable Unprintable>, 3.0]" in _text((None, error))

    def test_a_frame_size_keeps_the_dimensions_that_render(self) -> None:
        """Three components is the refusal, so the count must survive the rendering."""
        answer = video._frame_size_error([1.0, Unprintable(), 3.0])
        assert answer is not None
        assert "[1.0, <unrepresentable Unprintable>, 3.0]" in answer
