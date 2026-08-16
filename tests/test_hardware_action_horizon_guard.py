"""Behavior tests for the hardware control loop's accepted ``action_horizon`` domain.

``action_horizon`` is how many actions of each inferred chunk
``strands_robots.hardware_robot.Robot`` applies to the servo bus before
re-querying the policy: the task loop hands it to
:func:`~strands_robots.policies.base.resolve_chunk_length`, which coerces it with
``max(int(action_horizon), 1, ...)``. That coercion turned a caller mistake into a
plausible-but-different rollout instead of an error, and turned a non-numeric
value into a mid-task abort. These tests pin that a horizon the loop cannot
honor is refused at construction:

    - a ``0`` / negative horizon raises ``ValueError`` naming ``action_horizon``
      instead of being silently clamped to one action per inference - which
      re-queries an open-loop chunked checkpoint every step rather than replaying
      the chunk it was trained to emit;
    - ``2.7`` is refused instead of truncated, ``"4"`` instead of string-coerced,
      and ``True`` instead of acting as a silent horizon of 1;
    - ``None`` / ``nan`` / ``inf`` / a list are refused at construction rather
      than reaching ``int()`` only after the arm is connected and the first
      observation has been inferred on, aborting the task with a bare
      ``TypeError`` / ``ValueError``;
    - the refusal happens BEFORE the lerobot driver is built, so a rejected
      horizon never opens a serial port;
    - every horizon that IS accepted is the number of actions the loop applies
      per inference;
    - the accepted domain matches the simulation's rollout-count domain
      (``SimEngine._validate_positive_int``), so the same horizon cannot be
      refused for a digital twin and accepted for the arm it mirrors.

The ``bool`` half of that domain also closes a hole two callers previously worked
around locally (``SimEngine._validate_control_substeps`` rejected ``bool`` itself
before delegating, and the ``run_policy`` agent tool still rejects it for its own
required ``n_episodes``): a bare ``value < 1`` test lets ``True`` through as a
silent count of 1 while rejecting ``False``.

No serial/USB hardware is touched: ``_initialize_robot`` is stubbed with an
in-memory fake and the calibration migration is a no-op.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import positive_count_error
from tests.test_hardware_robot_lifecycle import _FakeLeRobot, _make_robot

# Horizons the loop cannot honor. ``0`` / negative collapse to a single action
# per inference; ``2.7`` truncates; ``"4"`` string-coerces; ``True`` acts as a
# silent 1; the rest are values ``int()`` cannot convert at all, so they abort
# the task only once the arm is already connected.
#
# ``8.0`` is refused for domain parity rather than because this loop could not
# use it: the simulation's rollout counts are consumed directly as ``range()``
# bounds where an integral float raises, and one shared rule is what keeps a
# horizon from being accepted here and refused for the matching simulated
# rollout. A caller holding a float from a config passes ``int(...)``.
UNUSABLE_HORIZONS: list[Any] = [
    0,
    -1,
    -8,
    2.7,
    8.0,
    "4",
    True,
    False,
    None,
    [4],
    float("nan"),
    float("inf"),
]

# Every positive integer is honorable, including the documented default.
USABLE_HORIZONS: list[Any] = [1, 2, 8, 64]


class _ChunkPolicy:
    """Emits a fixed 8-action chunk per inference, re-queried every action.

    ``execution_horizon`` of 1 makes ``action_horizon`` the sole decider of how
    much of each chunk the loop consumes (``resolve_chunk_length`` returns
    ``max(action_horizon, 1)``), so the applied-actions-per-inference ratio is
    exactly the horizon under test.
    """

    supports_rtc = False
    execution_horizon = 1
    actions_per_step = 1
    CHUNK = 8

    def __init__(self) -> None:
        self.inferences = 0

    def set_robot_state_keys(self, keys: list[str]) -> None:
        pass

    def set_control_frequency(self, hz: float) -> None:
        pass

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        pass

    async def get_actions(self, observation: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        self.inferences += 1
        return [{"j0.pos": float(i)} for i in range(self.CHUNK)]


@pytest.fixture
def hw_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct a real ``Robot.__init__`` without touching hardware.

    Returns a callable taking the keywords under test, carrying a ``built`` list
    that records every ``_initialize_robot`` call so a test can assert whether
    the lerobot driver was constructed.
    """
    built: list[Any] = []

    def fake_initialize_robot(self: HwRobot, robot: Any, cameras: Any, **kwargs: Any) -> _FakeLeRobot:
        built.append(robot)
        return _FakeLeRobot()

    monkeypatch.setattr(HwRobot, "_initialize_robot", fake_initialize_robot)
    monkeypatch.setattr(HwRobot, "_migrate_legacy_calibration", lambda self: None)

    def construct(**kwargs: Any) -> HwRobot:
        return HwRobot(tool_name="test_arm", robot="fake_arm", **kwargs)

    construct.built = built  # type: ignore[attr-defined]
    return construct


class TestUnusableHorizonRefused:
    @pytest.mark.parametrize("horizon", UNUSABLE_HORIZONS, ids=repr)
    def test_horizon_the_loop_cannot_honor_raises(self, hw_init, horizon):
        """A horizon with no honorable chunk length is refused, naming the parameter."""
        with pytest.raises(ValueError, match="action_horizon"):
            hw_init(action_horizon=horizon)

    @pytest.mark.parametrize("horizon", UNUSABLE_HORIZONS, ids=repr)
    def test_refusal_precedes_driver_construction(self, hw_init, horizon):
        """A rejected horizon never opens a serial port.

        The guard sits ahead of ``_initialize_robot``, which is what builds the
        lerobot driver and connects to the bus.
        """
        with pytest.raises(ValueError):
            hw_init(action_horizon=horizon)
        assert hw_init.built == []

    def test_message_reports_the_offending_value(self, hw_init):
        with pytest.raises(ValueError) as excinfo:
            hw_init(action_horizon=0)
        assert str(excinfo.value) == "Robot: action_horizon must be a positive integer, got 0."

    def test_a_valid_rate_does_not_excuse_an_invalid_horizon(self, hw_init):
        """The horizon is validated on its own, not only when siblings are bad."""
        with pytest.raises(ValueError, match="action_horizon"):
            hw_init(action_horizon=-4, control_frequency=50.0)


class TestUsableHorizonAccepted:
    @pytest.mark.parametrize("horizon", USABLE_HORIZONS, ids=repr)
    def test_accepted_horizon_is_kept_verbatim(self, hw_init, horizon):
        hw = hw_init(action_horizon=horizon)
        try:
            assert hw.action_horizon == horizon
            assert hw_init.built == ["fake_arm"]
        finally:
            hw.cleanup()

    def test_default_horizon_is_accepted(self, hw_init):
        hw = hw_init()
        try:
            assert hw.action_horizon == 8
        finally:
            hw.cleanup()


class TestAcceptedHorizonReachesTheLoop:
    """The horizon is the number of actions applied per inference.

    Nothing else pinned that ``action_horizon`` is what bounds the chunk slice
    the servo bus receives, which is the whole reason a wrong value matters.
    """

    @pytest.mark.parametrize("horizon", [1, 2, 4, 8], ids=repr)
    def test_actions_per_inference_equals_the_horizon(self, horizon):
        hw = _make_robot()
        hw.action_horizon = horizon
        policy = _ChunkPolicy()
        try:
            result = hw.run_policy(policy_object=policy, instruction="probe", n_steps=4 * horizon)
            assert result["status"] == "success", result
            assert len(hw.robot.sent_actions) == 4 * horizon
            assert policy.inferences == 4
        finally:
            hw.cleanup()


class TestHardwareAndSimulationShareOneDomain:
    """A count refused for a simulated rollout is refused for the real arm.

    The hardware constructor and the simulation's rollout-count guards bind the
    same :func:`~strands_robots.utils.positive_count_error` domain, so the two
    cannot drift apart on what counts as an honorable horizon.
    """

    @pytest.mark.parametrize("horizon", UNUSABLE_HORIZONS + USABLE_HORIZONS, ids=repr)
    def test_verdicts_match_the_simulation_rollout_guard(self, hw_init, horizon):
        sim_refuses = SimEngine._validate_action_horizon(horizon, "run_policy") is not None
        try:
            hw = hw_init(action_horizon=horizon)
        except ValueError:
            hardware_refuses = True
        else:
            hardware_refuses = False
            hw.cleanup()
        assert hardware_refuses == sim_refuses, (
            f"verdicts differ for action_horizon={horizon!r}: "
            f"hardware refuses={hardware_refuses}, simulation refuses={sim_refuses}"
        )

    def test_both_surfaces_report_the_same_wording(self, hw_init):
        with pytest.raises(ValueError) as excinfo:
            hw_init(action_horizon=-3)
        sim_text = SimEngine._validate_action_horizon(-3, "Robot")["content"][0]["text"]
        assert str(excinfo.value) == sim_text


class TestBoolIsOutsideTheSharedCountDomain:
    """``True`` cannot pass for a count of 1 on any surface.

    ``bool`` is an ``int`` subclass, so a bare ``value < 1`` test rejected
    ``False`` while letting ``True`` through - a value the caller never meant
    either way. Two callers worked around that locally rather than the shared
    guard enforcing it.
    """

    @pytest.mark.parametrize("param", ["n_episodes", "max_steps"])
    def test_rollout_counts_reject_true(self, param):
        result = SimEngine._validate_positive_int(True, param, "eval_policy")
        assert result is not None
        assert f"{param} must be a positive integer" in result["content"][0]["text"]

    def test_action_horizon_rejects_true(self):
        result = SimEngine._validate_action_horizon(True, "run_policy")
        assert result is not None
        assert "action_horizon must be a positive integer, got True." in result["content"][0]["text"]

    def test_control_substeps_still_rejects_true_after_delegating(self):
        """The local ``bool`` branch was removed; the shared domain covers it."""
        result = SimEngine._validate_control_substeps(True, "run_policy")
        assert result is not None
        assert result["content"][0]["text"] == "run_policy: control_substeps must be a positive integer, got True."

    def test_control_substeps_still_accepts_none(self):
        """``None`` means "auto-derive" and is the one documented exception."""
        assert SimEngine._validate_control_substeps(None, "run_policy") is None

    def test_shared_rule_rejects_both_bools(self):
        for value in (True, False):
            assert positive_count_error(value, "action_horizon", "Robot") is not None


class TestSharedCountDomainRejectsIntegralFloats:
    """The counts are ``range()`` bounds and slice indices, so only ``int`` works.

    This is what separates the rule from
    :func:`~strands_robots.utils.positive_whole_number_error`, whose media
    dimensions are arithmetic operands and may legitimately arrive as ``30.0``.
    """

    def test_integral_float_is_refused(self):
        assert positive_count_error(8.0, "action_horizon", "Robot") is not None

    def test_the_reason_is_real(self):
        with pytest.raises(TypeError):
            range(8.0)  # type: ignore[arg-type]
