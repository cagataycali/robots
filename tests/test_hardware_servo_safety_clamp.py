"""Behavior tests for the ``max_relative_target`` servo travel clamp.

``max_relative_target`` is the only per-command travel limit on the hardware
path: it caps how far each commanded goal position may move from the joint's
present position, so a value the driver cannot honor is a safety defect. The
config dataclass declares it as ``float | dict[str, float] | None`` and performs
no validation, while lerobot's consumer -
``lerobot.robots.utils.ensure_safe_goal_position`` - honors a narrower set of
values than that. The contracts pinned here:

    - a limit outside the honorable domain (non-finite, non-positive, ``bool``,
      non-numeric) is refused when the config is built, before the serial port
      is opened, instead of disabling the clamp or inverting it mid-rollout;
    - an ``int`` limit - type-correct against the field's annotation under PEP
      484's numeric tower - is normalized to ``float`` so it reaches the motors,
      rather than raising ``TypeError`` at the first servo command;
    - the same domain applies to every value of a per-motor mapping;
    - ``None`` still means "clamp disabled", the field's documented default;
    - every limit this layer accepts really does clamp an over-large goal when
      handed to lerobot's own consumer - the parity that keeps the two domains
      from drifting apart again.

No serial/USB hardware is touched: only config dataclasses are constructed, and
``make_robot_from_config`` is stubbed for the driver-construction branch.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

pytest.importorskip("lerobot")

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState

ROBOT_TYPE = "so101_follower"

# Limits no driver can honor, with what each one does when it reaches the bus.
UNHONORABLE = [
    pytest.param(float("nan"), id="nan-disables-the-clamp"),
    pytest.param(float("inf"), id="inf-disables-the-clamp"),
    pytest.param(-5.0, id="negative-inverts-the-clamp"),
    pytest.param(-1, id="negative-int"),
    pytest.param(0.0, id="zero-discards-every-motion"),
    pytest.param(0, id="zero-int"),
    pytest.param(True, id="bool-acts-as-one-tick"),
    pytest.param(False, id="bool-false"),
    pytest.param("5", id="numeric-looking-string"),
    pytest.param([5.0], id="list"),
]


def _make_robot() -> HwRobot:
    """A Robot wired with just the attributes ``_create_minimal_config`` /
    ``_initialize_robot`` need, plus the handful the destructor's cleanup path
    reads (so teardown is silent) - never touching hardware."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw.mesh = None
    return hw


def _build(limit: Any) -> Any:
    """Build the real lerobot config through the strands forwarding path."""
    return _make_robot()._create_minimal_config(ROBOT_TYPE, None, port="/dev/null", max_relative_target=limit)


def _clamp_verdict(limit: Any) -> str:
    """How ``limit`` behaves when the driver applies it to a servo command.

    Drives lerobot's own ``ensure_safe_goal_position`` with an over-large move
    in BOTH directions - an inverted limit only shows itself on a move toward a
    goal below the present position - and reports the first way it fails to be
    a travel clamp:

        - ``"raises"``: refused outright at the first servo command;
        - ``"unbounded"``: the full requested travel is applied, so the limit
          is not limiting anything;
        - ``"wrong-direction"``: the command moves away from the goal;
        - ``"frozen"``: no motion is permitted at all;
        - ``"clamps"``: bounded, toward the goal, and non-zero - a real clamp.
    """
    from lerobot.robots.utils import ensure_safe_goal_position

    present = 0.0
    for goal in (1000.0, -1000.0):
        try:
            safe = ensure_safe_goal_position({"shoulder_pan": (goal, present)}, limit)["shoulder_pan"]
        except (TypeError, ValueError):
            return "raises"
        travel = safe - present
        requested = goal - present
        if not math.isfinite(safe) or abs(travel) >= abs(requested):
            return "unbounded"
        if travel == 0.0:
            return "frozen"
        if (travel > 0) != (requested > 0):
            return "wrong-direction"
    return "clamps"


class TestUnhonorableLimitIsRefused:
    @pytest.mark.parametrize("limit", UNHONORABLE)
    def test_scalar_limit_outside_the_domain_is_refused(self, limit):
        """A limit the driver cannot honor is refused, naming the parameter.

        Pre-fix every one of these built a config: ``nan``/``inf`` silently
        disabled the clamp, a negative limit turned it into a fixed-magnitude
        step generator, ``0`` froze the arm while the rollout reported success,
        and ``True``/``"5"``/``[5.0]`` surfaced a bare ``TypeError(True)`` at the
        first servo command with no parameter name in sight.
        """
        with pytest.raises(ValueError, match="max_relative_target"):
            _build(limit)

    @pytest.mark.parametrize("limit", UNHONORABLE)
    def test_per_motor_limit_outside_the_domain_is_refused(self, limit):
        """A mapping value gets the same domain as a scalar, named per motor."""
        with pytest.raises(ValueError, match=r"max_relative_target\['shoulder_pan'\]"):
            _build({"shoulder_pan": limit})

    def test_empty_mapping_is_refused(self):
        """An empty per-motor mapping can never match the driver's motor set."""
        with pytest.raises(ValueError, match="must not be an empty mapping"):
            _build({})

    def test_non_name_mapping_key_is_refused(self):
        """A key that is not a motor name can never match one."""
        with pytest.raises(ValueError, match="keys must be motor names"):
            _build({1: 5.0})

    def test_refusal_precedes_driver_construction(self, monkeypatch):
        """The limit is refused before lerobot builds a driver for the port.

        Guard placement matters: a value rejected only once the driver exists
        has already opened the serial port and energized the bus.
        """
        import lerobot.robots.utils as lru

        made: list[Any] = []
        monkeypatch.setattr(lru, "make_robot_from_config", lambda config: made.append(config))

        hw = _make_robot()
        with pytest.raises(ValueError, match="max_relative_target"):
            hw._initialize_robot(ROBOT_TYPE, None, port="/dev/null", max_relative_target=float("nan"))
        assert made == []


class TestHonorableLimitReachesTheMotors:
    def test_int_limit_is_normalized_so_it_clamps(self):
        """An ``int`` limit is normalized to ``float`` and really does clamp.

        The field is annotated ``float | dict[str, float] | None`` and PEP 484's
        numeric tower makes an ``int`` assignable to it, so ``10`` type-checks -
        but lerobot's consumer dispatches on ``isinstance(value, float)`` and
        pre-fix raised ``TypeError(10)`` on every servo command.
        """
        from lerobot.robots.utils import ensure_safe_goal_position

        cfg = _build(5)
        assert cfg.max_relative_target == 5.0
        assert isinstance(cfg.max_relative_target, float)
        # Goal 40 from a present 0 exceeds the 5-tick limit, so it is capped.
        assert ensure_safe_goal_position({"shoulder_pan": (40.0, 0.0)}, cfg.max_relative_target) == {
            "shoulder_pan": 5.0
        }

    def test_per_motor_int_limits_are_normalized(self):
        cfg = _build({"shoulder_pan": 5, "wrist_flex": 2.5})
        assert cfg.max_relative_target == {"shoulder_pan": 5.0, "wrist_flex": 2.5}
        assert all(isinstance(v, float) for v in cfg.max_relative_target.values())

    def test_none_still_means_the_clamp_is_disabled(self):
        """``None`` is the field's documented default and stays a valid value."""
        assert _build(None).max_relative_target is None

    def test_omitting_the_limit_leaves_the_field_at_its_default(self):
        cfg = _make_robot()._create_minimal_config(ROBOT_TYPE, None, port="/dev/null")
        assert cfg.max_relative_target is None


class TestDomainMatchesTheDriverConsumer:
    """Parity: what this layer accepts is exactly what the driver can honor.

    Without this the two domains drift - which is how a value the config
    dataclass accepts came to be one the consumer refuses.
    """

    @pytest.mark.parametrize(
        "limit",
        [
            pytest.param(5.0, id="float"),
            pytest.param(5, id="int"),
            pytest.param(2.5, id="fractional"),
            pytest.param({"shoulder_pan": 5}, id="mapping-of-int"),
            pytest.param({"shoulder_pan": 2.5}, id="mapping-of-float"),
        ],
    )
    def test_every_accepted_limit_behaves_as_a_clamp(self, limit):
        assert _clamp_verdict(_build(limit).max_relative_target) == "clamps"

    @pytest.mark.parametrize("limit", UNHONORABLE)
    def test_every_refused_limit_would_not_have_behaved_as_a_clamp(self, limit):
        """Each refused limit misbehaves on the wire, in its own way.

        This is the other half of the parity: the guard is not merely stricter
        than the consumer, it refuses exactly the values whose outcome at the
        motors is wrong.
        """
        assert _clamp_verdict(limit) != "clamps"
