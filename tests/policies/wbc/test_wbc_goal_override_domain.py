"""Domain contracts for the WBC per-call goal overrides ``height`` / ``target_orientation``.

``WBCPolicy.get_actions`` reads its locomotion goal from well-known kwargs. Three
of them shape the command block the network sees, and until now only one had a
domain:

* ``target_velocity`` -> ``_validate_velocity``, refused non-finite (already).
* ``height`` -> ``command[3]``, unchecked.
* ``target_orientation`` -> ``command[4:7]`` (``[5:8]`` on the gait variant),
  unchecked.

The two unchecked ones are the PER-CALL SPELLING of two config fields #1991 gave
a domain to. ``height`` overrides :attr:`WBCConfig.height_cmd` and
``target_orientation`` overrides :attr:`WBCConfig.rpy_cmd`; they are written into
the same command slots, and when a call omits them those config fields supply the
value. So the same number was refused as a default and accepted as an override of
that default - the inversion these tests exist to close.

Both are APPLIED rather than forwarded: the command block is a slice of the
network's own input, so nothing downstream refuses an unusable value. It becomes
a plausible-looking chunk of joint targets instead of an error, and it does not
stay confined to the tick that caused it - see
:class:`TestWhyTheGoalOverrideDomainIsWhatItIs`, which drives the real
``get_actions`` and measures the damage rather than asserting it.

The tests are dependency-free: ``allow_missing_models=True`` is the documented
seam for injecting a stub session, and the observation/command path needs only
numpy, so neither ``onnxruntime`` nor ``mujoco`` is required.
"""

from __future__ import annotations

import ast
import asyncio
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCConfig, WBCPolicy
from strands_robots.policies.wbc.gait import GAIT_COMMAND_DIM, GAIT_SINGLE_OBS_DIM, WBCGaitPolicy
from strands_robots.utils import finite_number_error, finite_vector_error

# ---------------------------------------------------------------------------
# Probe values. Every entry is refused by WBCConfig for the field the per-call
# kwarg overrides, which is what makes the parity class below non-trivial.
# ---------------------------------------------------------------------------

# ``bool`` is in both sets because ``float(True)`` is a silent ``1.0`` - a real
# base-height / roll command, not an error - and a numeric string because it is
# the shape a value arriving from a config file or an LLM tool call takes.
_UNUSABLE_SCALARS: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.8",
    [0.8],  # a list where a scalar belongs
]

_UNUSABLE_COMPONENTS: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "0.1",
    [0.1],  # a nested list where a component belongs
]


# ---------------------------------------------------------------------------
# Dependency-free doubles
# ---------------------------------------------------------------------------


class _Input:
    name = "obs"


class _PropagatingSession:
    """Stand-in for ``onnxruntime.InferenceSession`` that propagates non-finites.

    Models the one property of the real network that decides this domain: the
    first matmul of a dense policy touches EVERY element of its input, so a
    single non-finite input makes every output non-finite. ``fill + 0.0 *
    sum(obs)`` is the minimal faithful stand-in - it returns exactly ``fill`` for
    any finite observation, so the control rows are unaffected by the modelling
    choice, and ``nan``/``inf`` propagate as they would through a real matmul.
    """

    def __init__(self, num_actions: int, fill: float = 0.04) -> None:
        self.num_actions = num_actions
        self.fill = fill
        self.calls = 0

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _output_names: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        self.calls += 1
        (arr,) = feed.values()
        taint = 0.0 * float(np.sum(np.asarray(arr, dtype=np.float64)))
        return [np.full((1, self.num_actions), self.fill + taint, dtype=np.float32)]


def _config(**overrides: Any) -> WBCConfig:
    base: dict[str, Any] = dict(policy_path="policy.onnx", obs_history_len=1)
    base.update(overrides)
    return WBCConfig(**base)


def _gait_config(**overrides: Any) -> WBCConfig:
    base: dict[str, Any] = dict(
        policy_path="policy.onnx",
        single_obs_dim=GAIT_SINGLE_OBS_DIM,
        command_dim=GAIT_COMMAND_DIM,
        obs_history_len=1,
    )
    base.update(overrides)
    return WBCConfig(**base)


def _policy(**overrides: Any) -> WBCPolicy:
    return WBCPolicy(config=_config(**overrides), allow_missing_models=True)


def _gait_policy(**overrides: Any) -> WBCGaitPolicy:
    return WBCGaitPolicy(config=_gait_config(**overrides), allow_missing_models=True)


def _running_policy() -> tuple[WBCPolicy, _PropagatingSession]:
    """A policy wired end to end so ``get_actions`` can actually be driven."""
    p = _policy()
    session = _PropagatingSession(p._config.num_actions)
    p.policy_session = session
    p.walk_session = session
    p.set_robot_state_keys(["floating_base_joint", *WBC_G1_ALL_JOINTS])
    return p, session


def _observation() -> dict[str, Any]:
    return {
        "joint_positions": {n: 0.0 for n in WBC_G1_ALL_JOINTS},
        "joint_velocities": {n: 0.0 for n in WBC_G1_ALL_JOINTS},
        "base_angular_velocity": [0.0, 0.0, 0.0],
        "base_orientation": [1.0, 0.0, 0.0, 0.0],
    }


def _tick(policy: WBCPolicy, **goal: Any) -> np.ndarray:
    """Drive one real ``get_actions`` and return the joint targets it produced."""
    actions = asyncio.run(policy.get_actions(_observation(), "", target_velocity=[0.1, 0.0, 0.0], **goal))
    return np.asarray(list(actions[0].values()), dtype=np.float64)


# ---------------------------------------------------------------------------
# Why the domain is what it is - measured on the real conversion, not asserted
# ---------------------------------------------------------------------------


class TestWhyTheGoalOverrideDomainIsWhatItIs:
    """Drive the real ``get_actions`` and measure what each refused value did.

    These are the behavioural justification for the guard: without them the
    domain is an unexplained restriction, and a later reader cannot tell which
    bound is load-bearing.
    """

    def test_a_usable_goal_drives_finite_targets_on_every_tick(self) -> None:
        p, _ = _running_policy()
        for _ in range(5):
            targets = _tick(p, height=0.8, target_orientation=[0.1, 0.2, 0.3])
            assert np.all(np.isfinite(targets))

    def test_a_non_finite_height_would_have_made_every_joint_target_non_finite(self) -> None:
        p, _ = _running_policy()
        command, _ = p._resolve_command({"target_velocity": [0.1, 0.0, 0.0]})
        command[3] = float("nan")  # what the unguarded path wrote
        assert not math.isfinite(command[3])
        # And the network's response to such a command block is wholly non-finite.
        session = _PropagatingSession(p._config.num_actions)
        poisoned = session.run(None, {"obs": np.concatenate([command, np.zeros(79)])})
        assert not np.any(np.isfinite(np.asarray(poisoned[0])))

    def test_one_unusable_height_would_have_poisoned_every_later_tick(self) -> None:
        """The sharpest row: the damage outlives the tick that caused it.

        ``_prev_action`` is fed into every subsequent observation frame, so a
        single non-finite action makes every later action non-finite too - with
        the caller's next, perfectly usable goal applied exactly as asked, and no
        error at any point. Measured by writing the poisoned action the unguarded
        path would have produced, then driving usable ticks.
        """
        p, _ = _running_policy()
        p._prev_action = np.full(p._config.num_actions, float("nan"))
        for _ in range(4):
            targets = _tick(p, height=0.8)  # a usable goal
            assert not np.any(np.isfinite(targets)), "a usable goal should not have recovered"

    def test_only_reset_would_have_cleared_it(self) -> None:
        p, _ = _running_policy()
        p._prev_action = np.full(p._config.num_actions, float("nan"))
        p.reset()
        assert np.all(np.isfinite(_tick(p, height=0.8)))

    def test_a_bool_height_would_have_installed_a_silent_one_metre_command(self) -> None:
        """``float(True)`` is ``1.0`` - a real base-height command, not an error."""
        assert float(True) == 1.0
        assert finite_number_error(True, "height", "WBCPolicy.get_actions") is not None

    def test_a_string_height_would_have_been_a_numeric_string_read_as_a_command(self) -> None:
        assert float("0.8") == 0.8
        assert finite_number_error("0.8", "height", "WBCPolicy.get_actions") is not None

    def test_a_non_finite_orientation_component_reaches_the_command_block(self) -> None:
        p = _policy()
        command, _ = p._resolve_command({"target_velocity": [0.0, 0.0, 0.0]})
        command[4:7] = [float("nan"), 0.0, 0.0]  # what the unguarded path wrote
        assert not np.all(np.isfinite(np.asarray(command[4:7])))


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


class TestHeightOverrideDomain:
    @pytest.mark.parametrize("value", _UNUSABLE_SCALARS)
    def test_an_unusable_height_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="height"):
            _policy()._resolve_command({"height": value})

    def test_the_refusal_names_the_parameter_the_method_and_the_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _policy()._resolve_command({"height": float("nan")})
        message = str(excinfo.value)
        assert "height" in message
        assert "WBCPolicy.get_actions" in message
        assert "nan" in message

    @pytest.mark.parametrize("value", [0.74, 0.8, 0.0, -0.25, 2.5, np.float64(0.8), 1])
    def test_a_usable_height_is_accepted_and_honored(self, value: Any) -> None:
        command, _ = _policy()._resolve_command({"height": value})
        assert command[3] == pytest.approx(float(value))

    def test_it_is_refused_through_the_public_get_actions_too(self) -> None:
        p, _ = _running_policy()
        with pytest.raises(ValueError, match="height"):
            _tick(p, height=float("nan"))


class TestTargetOrientationOverrideDomain:
    @pytest.mark.parametrize("component", _UNUSABLE_COMPONENTS)
    def test_an_unusable_orientation_component_is_refused(self, component: Any) -> None:
        with pytest.raises(ValueError, match="target_orientation"):
            _policy()._resolve_command({"target_orientation": [component, 0.0, 0.0]})

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_every_component_is_examined_not_just_the_first(self, index: int) -> None:
        rpy = [0.0, 0.0, 0.0]
        rpy[index] = float("nan")
        with pytest.raises(ValueError, match="target_orientation"):
            _policy()._resolve_command({"target_orientation": rpy})

    def test_a_component_beyond_the_command_block_is_still_examined(self) -> None:
        """A 7-wide command consumes rpy[0:3]; a bad rpy[3] is truncated away.

        It is still refused, because the caller asked for it. Accepting a value
        purely because this ``command_dim`` happens to discard it would make the
        domain depend on the config, so the same call would be refused or
        accepted depending on a width the caller never mentioned.
        """
        with pytest.raises(ValueError, match="target_orientation"):
            _policy()._resolve_command({"target_orientation": [0.0, 0.0, 0.0, float("nan")]})

    def test_a_scalar_orientation_is_refused_naming_the_parameter(self) -> None:
        """Parity, not over-reach: the config field refuses a scalar too.

        ``WBCConfig(rpy_cmd=0.5)`` raises ``TypeError: 'float' object is not
        iterable`` out of the component loop - a refusal that names neither the
        field nor the value. So both surfaces refuse a scalar orientation; this
        one now says which parameter and why.
        """
        with pytest.raises(ValueError, match="target_orientation"):
            _policy()._resolve_command({"target_orientation": 0.5})
        with pytest.raises((TypeError, ValueError)):
            WBCConfig(policy_path="p.onnx", rpy_cmd=0.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "rpy",
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
            (0.1, 0.2, 0.3),
            np.array([0.1, 0.2, 0.3]),
            [np.float64(0.1), np.float64(0.2), np.float64(0.3)],
        ],
    )
    def test_a_usable_orientation_is_accepted_and_honored(self, rpy: Any) -> None:
        command, _ = _policy()._resolve_command({"target_orientation": rpy})
        assert np.allclose(np.asarray(command[4:7], dtype=np.float64), np.asarray(rpy, dtype=np.float64)[:3])

    def test_it_is_refused_through_the_public_get_actions_too(self) -> None:
        p, _ = _running_policy()
        with pytest.raises(ValueError, match="target_orientation"):
            _tick(p, target_orientation=[float("inf"), 0.0, 0.0])


# ---------------------------------------------------------------------------
# The gait variant overrides _resolve_command wholesale, so it needs its own
# coverage - a guard in the base class would not have reached it.
# ---------------------------------------------------------------------------


class TestTheGaitVariantSharesTheDomain:
    @pytest.mark.parametrize("value", _UNUSABLE_SCALARS)
    def test_an_unusable_gait_height_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="height"):
            _gait_policy()._resolve_command({"height": value})

    @pytest.mark.parametrize("component", _UNUSABLE_COMPONENTS)
    def test_an_unusable_gait_orientation_component_is_refused(self, component: Any) -> None:
        with pytest.raises(ValueError, match="target_orientation"):
            _gait_policy()._resolve_command({"target_orientation": [component, 0.0, 0.0]})

    def test_the_gait_refusal_names_the_gait_class_not_the_base(self) -> None:
        """The message must name the class the caller actually used."""
        with pytest.raises(ValueError) as excinfo:
            _gait_policy()._resolve_command({"height": float("nan")})
        assert "WBCGaitPolicy.get_actions" in str(excinfo.value)

    def test_the_gait_slots_still_receive_a_usable_goal(self) -> None:
        command, _ = _gait_policy()._resolve_command({"height": 0.8, "target_orientation": [0.1, 0.2, 0.3]})
        assert command[3] == pytest.approx(0.8)
        assert np.allclose(np.asarray(command[5:8], dtype=np.float64), [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Cross-surface parity - the property the whole change exists to restore
# ---------------------------------------------------------------------------


class TestTheTwoSurfacesForOneCommandSlotAgree:
    """Neither surface writing a command slot may accept what the other refuses.

    ``height_cmd``/``height`` and ``rpy_cmd``/``target_orientation`` are two
    spellings of one value: whichever is supplied lands in the same slot. A value
    the config refuses as a default but the override accepts (or the reverse) is
    the defect, independent of which domain is the "right" one.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, "0.8"])
    def test_a_height_neither_surface_may_accept(self, value: Any) -> None:
        with pytest.raises(ValueError):
            WBCConfig(policy_path="p.onnx", height_cmd=value)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            _policy()._resolve_command({"height": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, "0.1"])
    def test_an_orientation_component_neither_surface_may_accept(self, value: Any) -> None:
        with pytest.raises(ValueError):
            WBCConfig(policy_path="p.onnx", rpy_cmd=[value, 0.0, 0.0])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            _policy()._resolve_command({"target_orientation": [value, 0.0, 0.0]})

    @pytest.mark.parametrize("value", [0.0, 0.74, -0.25, 2.5])
    def test_a_height_both_surfaces_must_accept(self, value: float) -> None:
        assert WBCConfig(policy_path="p.onnx", height_cmd=value).height_cmd == pytest.approx(value)
        command, _ = _policy()._resolve_command({"height": value})
        assert command[3] == pytest.approx(value)

    def test_the_per_call_guard_is_the_shared_domain_the_config_uses(self) -> None:
        """Same helper, so the two cannot drift apart by editing one of them."""
        for value in [float("nan"), float("inf"), True, "0.8", 0.8, 0.0, -1.0]:
            assert (finite_number_error(value, "height", "ctx") is None) == (
                finite_number_error(value, "height_cmd", "WBCConfig") is None
            )


# ---------------------------------------------------------------------------
# Guard placement - a refused goal must apply nothing at all
# ---------------------------------------------------------------------------


class TestNothingIsAppliedByARefusedGoal:
    """A refused goal must leave the policy exactly as it was.

    ``_resolve_command`` is the first statement of ``get_actions`` and the guard
    is the first statement of ``_resolve_command``, so a refusal precedes every
    mutation on the tick: the observation frame is not pushed, ``_prev_action``
    is not replaced, the gait phase is not advanced and the session is not run.
    Placing the guard here rather than at the point of use is what removes the
    need for any undo path.
    """

    def test_a_refused_goal_runs_no_inference(self) -> None:
        p, session = _running_policy()
        with pytest.raises(ValueError):
            _tick(p, height=float("nan"))
        assert session.calls == 0

    def test_a_refused_goal_pushes_no_observation_frame(self) -> None:
        p, _ = _running_policy()
        before = len(p._history._frames) if hasattr(p._history, "_frames") else None
        with pytest.raises(ValueError):
            _tick(p, target_orientation=[float("nan"), 0.0, 0.0])
        after = len(p._history._frames) if hasattr(p._history, "_frames") else None
        assert after == before

    def test_a_refused_goal_leaves_prev_action_untouched(self) -> None:
        p, _ = _running_policy()
        marker = np.full(p._config.num_actions, 0.123)
        p._prev_action = marker
        with pytest.raises(ValueError):
            _tick(p, height=float("inf"))
        assert p._prev_action is marker

    def test_a_refused_goal_does_not_advance_the_gait_phase(self) -> None:
        """The gait clock is stateful, so a half-applied tick would be visible."""
        g = _gait_policy()
        before_indices = g._gait_clock.gait_indices
        before_started = g._gait_clock.just_started
        with pytest.raises(ValueError):
            g._resolve_command({"height": float("nan")})
        assert g._gait_clock.gait_indices == before_indices
        assert g._gait_clock.just_started == before_started

    def test_the_next_usable_tick_behaves_as_if_the_refusal_never_happened(self) -> None:
        p, session = _running_policy()
        reference, _ = _running_policy()
        with pytest.raises(ValueError):
            _tick(p, height=float("nan"))
        assert np.allclose(_tick(p, height=0.8), _tick(reference, height=0.8))
        assert session.calls == 1


# ---------------------------------------------------------------------------
# Over-reach controls - what the guard must keep accepting
# ---------------------------------------------------------------------------


class TestTheUsableGoalsStayFirstClass:
    def test_an_omitted_height_still_takes_the_config_default(self) -> None:
        command, _ = _policy(height_cmd=0.71)._resolve_command({})
        assert command[3] == pytest.approx(0.71)

    def test_an_explicit_none_height_still_means_no_override(self) -> None:
        command, _ = _policy(height_cmd=0.71)._resolve_command({"height": None})
        assert command[3] == pytest.approx(0.71)

    def test_an_explicit_none_orientation_still_means_no_override(self) -> None:
        command, _ = _policy(rpy_cmd=[0.05, 0.0, 0.0])._resolve_command({"target_orientation": None})
        assert np.allclose(np.asarray(command[4:7], dtype=np.float64), [0.05, 0.0, 0.0])

    def test_a_long_orientation_is_still_truncated_rather_than_refused(self) -> None:
        """REGRESSION guard: a 6-component orientation was already supported."""
        command, _ = _policy()._resolve_command({"target_orientation": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
        assert np.allclose(np.asarray(command[4:7], dtype=np.float64), [1.0, 0.0, 0.0])

    def test_a_short_orientation_is_still_accepted(self) -> None:
        command, _ = _policy()._resolve_command({"target_orientation": [0.1]})
        assert command[4] == pytest.approx(0.1)

    def test_an_empty_orientation_is_still_accepted_as_no_component(self) -> None:
        command, _ = _policy()._resolve_command({"target_orientation": []})
        assert np.allclose(np.asarray(command[4:7], dtype=np.float64), [0.0, 0.0, 0.0])

    def test_a_usable_goal_pair_still_drives_get_actions_end_to_end(self) -> None:
        p, session = _running_policy()
        targets = _tick(p, height=0.8, target_orientation=[0.1, 0.2, 0.3])
        assert len(targets) == p._config.num_actions
        assert np.all(np.isfinite(targets))
        assert session.calls == 1

    def test_target_velocity_keeps_its_own_existing_domain(self) -> None:
        """Deliberately untouched: it already had a guard, with its own message.

        ``_validate_velocity`` refuses non-finite components and a short vector.
        It is NARROWER than the shared domain used here - it accepts ``True`` and
        ``"0.5"``, which numpy coerces - but it is self-consistent across both
        surfaces that read it (the constructor default and the per-call kwarg),
        so widening it is a separate concern with its own callers to survey and
        is not folded into this change.
        """
        with pytest.raises(ValueError, match="target_velocity"):
            _policy()._resolve_command({"target_velocity": [float("nan"), 0.0, 0.0]})
        command, _ = _policy()._resolve_command({"target_velocity": [0.5, 0.0, 0.0]})
        assert np.all(np.isfinite(np.asarray(command[:3], dtype=np.float64)))


# ---------------------------------------------------------------------------
# Structural guard - a third _resolve_command cannot skip the domain
# ---------------------------------------------------------------------------

_WBC_PACKAGE = Path(WBCPolicy.__module__.replace(".", "/")).parent
_GUARD_CALL = "_validate_goal_overrides"
_GOAL_KWARGS = ("height", "target_orientation")

# Every ``_resolve_command`` that exists today. Listed exactly so the scan cannot
# pass by finding nothing (the non-vacuity failure a structural test is most
# prone to), and so adding a surface is a deliberate edit of this set.
_EXPECTED_SURFACES = {("policy.py", "_resolve_command"), ("gait.py", "_resolve_command")}


def _wbc_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[3] / "strands_robots" / "policies" / "wbc"
    return sorted(p for p in root.glob("*.py") if p.name != "__init__.py")


def _goal_reading_functions(source: str) -> list[tuple[str, bool]]:
    """Return ``(function name, routes through the guard)`` for goal readers.

    A function "reads a goal" when it names one of the goal kwargs as a string
    literal, which is how both ``_resolve_command`` implementations reach them
    (``kwargs.get("height")``).
    """
    found: list[tuple[str, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # The guard itself names both kwargs, and its own name appears in its
        # dump, so counting it would make it trivially "guarded" and hide it
        # among its consumers. It IS the domain; the scan is about who routes
        # through it.
        if node.name == _GUARD_CALL:
            continue
        body = ast.dump(node)
        reads_goal = any(f"value='{kwarg}'" in body for kwarg in _GOAL_KWARGS)
        if reads_goal:
            found.append((node.name, _GUARD_CALL in body))
    return found


class TestNoGoalOverrideSurfaceDrifts:
    def test_every_goal_reading_surface_routes_through_the_guard(self) -> None:
        adrift = []
        for path in _wbc_sources():
            for name, guarded in _goal_reading_functions(path.read_text()):
                if not guarded:
                    adrift.append((path.name, name))
        assert not adrift, f"these read a WBC goal override without a domain: {sorted(adrift)}"

    def test_the_scan_finds_the_surfaces_it_is_supposed_to_find(self) -> None:
        """Non-vacuity: a scan that matches nothing would pass the test above."""
        found = {(path.name, name) for path in _wbc_sources() for name, _ in _goal_reading_functions(path.read_text())}
        assert found == _EXPECTED_SURFACES, (
            f"the set of goal-reading surfaces changed: {sorted(found)}; "
            "add the domain to any new one, then update _EXPECTED_SURFACES"
        )

    def test_a_planted_unguarded_surface_is_caught(self) -> None:
        """Meta-test: the scan must fail on a surface that skips the guard."""
        planted = "def _resolve_command(self, kwargs):\n    return kwargs.get('height')\n"
        assert _goal_reading_functions(planted) == [("_resolve_command", False)]

    def test_a_planted_guarded_surface_is_accepted(self) -> None:
        planted = (
            "def _resolve_command(self, kwargs):\n"
            "    self._validate_goal_overrides(kwargs)\n"
            "    return kwargs.get('height')\n"
        )
        assert _goal_reading_functions(planted) == [("_resolve_command", True)]


class TestTheGuardIsTheSharedDomainAndNothingLocal:
    """The guard must add no rule of its own beyond delegating to the helpers."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, False, "0.8", 0.0, 0.8, -1.0, 2.5])
    def test_the_height_verdict_is_exactly_finite_number_error(self, value: Any) -> None:
        expected_refusal = finite_number_error(value, "height", "WBCPolicy.get_actions") is not None
        try:
            _policy()._resolve_command({"height": value})
            refused = False
        except ValueError:
            refused = True
        assert refused == expected_refusal

    @pytest.mark.parametrize(
        "rpy", [[float("nan"), 0.0, 0.0], [True, 0.0, 0.0], ["a", 0.0, 0.0], [0.1, 0.2, 0.3], [], 0.5]
    )
    def test_the_orientation_verdict_is_exactly_finite_vector_error(self, rpy: Any) -> None:
        expected_refusal = finite_vector_error("WBCPolicy.get_actions", "target_orientation", rpy) is not None
        try:
            _policy()._resolve_command({"target_orientation": rpy})
            refused = False
        except ValueError:
            refused = True
        assert refused == expected_refusal
