"""The WBC providers discard the ``instruction`` string - pinned, not just stated.

:meth:`WBCPolicy.get_actions` and :meth:`WBCGaitPolicy.get_actions` each accept a
natural-language ``instruction`` and read no goal from it: the locomotion command
comes only from the well-known kwargs (``target_velocity``, ``target_orientation``,
``height``, and the gait variant's ``gait_frequency``). Both docstrings say so
("``instruction`` is ignored"). Neither said it in a test.

Why the gap matters more here than on a kwargs-only provider
------------------------------------------------------------
On the WBC providers the instruction is not a parameter a caller has to go out of
their way to reach - on the hardware path it is the ONLY thing a caller supplies.
:meth:`HardwareRobot._run_policy_loop` calls::

    robot_actions = await policy_instance.get_actions(observation, instruction)

with no kwargs at all, and the mesh dispatcher that feeds it *requires* a
non-empty one: ``validate_command`` refuses an ``execute``/``start`` payload
without it ("execute/start requires non-empty `instruction`"), and ``wbc`` is on
the ``is_safe_policy_provider`` allowlist that the same validator enforces - a
pairing ``tests/policies/wbc/test_policy.py::TestMeshSecurityAllowlist`` already
pins with the literal command ``{"action": "execute", "instruction": "walk
forward", "policy_provider": "wbc", ...}``.

So a peer drives WBC by sending words, the words are mandatory, and every goal
component is then taken from the config instead. "walk forward" is accepted and
the robot does whatever ``target_velocity`` defaults to, with ``status="success"``
and no signal that the request was dropped.

Why the claim held vacuously
----------------------------
Across the whole WBC suite no ``get_actions``/``get_actions_sync`` call passed a
non-empty instruction - the one occurrence of a goal-bearing string is the mesh
*command dict* above, which never reaches a policy. So wiring the instruction
into the goal resolution would have turned both docstrings into lies with every
test still green: the failure mode is a silently-wrong doc, not a red build.

What is NOT proposed
--------------------
Refusing a non-empty instruction. Every :class:`Policy` accepts one and the
runner forwards it verbatim to the providers that DO read it (``curobo`` parses a
JSON goal out of it; see ``tests/policies/curobo/test_instruction_goal_extraction.py``),
so refusing here would break the uniform policy interface to fix a visibility
problem. The discard is correct; it just needed enforcing rather than describing.
Acceptance of a non-empty instruction is therefore pinned too - every probe below
completes and returns a well-formed action dict.

What is pinned
--------------
For each provider, one tick on a FRESH policy per probe instruction, asserting
the instruction reaches neither of the two channels it could plausibly steer:

1. the observation array handed to the ONNX session - byte-identical, which
   covers the command block (velocity, height, roll/pitch/yaw) and the gait
   variant's step-frequency slot and clock tail in one comparison;
2. the walk-vs-main session choice on :class:`WBCPolicy` - a "stand still"
   instruction must not move the tick off the walk policy.

plus the returned action dict, compared exactly (plain floats, so no tolerance).

Non-vacuity
-----------
An assertion that nothing changed is exactly where a test can pass by measuring
nothing, so :class:`TestTheMeasuredChannelsAreLive` drives the same probe values
through the documented kwarg spellings and asserts each channel DOES move: the
fed observation changes for a different velocity, orientation and height, and a
zero velocity moves the tick to the main session. Two of the probe strings are
also written in spellings a naive wiring would parse - a JSON object like the one
``curobo`` accepts, and a ``key=value`` pair - so they are not inert text.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBCGaitPolicy

from .test_gait import _full_g1_obs
from .test_policy import _g1_keys, _make_policy, _StubSession

# Instructions that state a goal in words. The first is the spelling every
# pre-existing WBC call used, so it is the baseline the rest are compared to; the
# empty string is what made the contract vacuous.
_BASELINE_INSTRUCTION = ""

_GOAL_BEARING_INSTRUCTIONS: list[str] = [
    "walk forward",
    "walk backward twice as fast",
    "turn left",
    # Would flip the walk-vs-main session choice if it were read at all.
    "stand still, do not move",
    # A spelling curobo really does parse a goal out of, so copying that
    # behaviour onto WBC is a realistic regression rather than a hypothetical.
    '{"target_velocity": [1.0, 0.0, 0.0]}',
    "target_orientation=[0.4, 0.0, 0.0]",
    # Whitespace-only: not empty, so it takes the same path as the rest.
    "   \n\t  ",
]

# A non-trivial goal supplied the documented way, held fixed across every probe:
# non-zero velocity (so the walk session is the one that runs), plus an
# orientation and height that put values in the command block's other slots.
_GOAL_KWARGS: dict[str, Any] = {
    "target_velocity": [0.5, -0.2, 0.3],
    "target_orientation": [0.1, 0.2, 0.3],
    "height": 0.74,
}


def _tick_wbc(instruction: str, **goal: Any) -> tuple[np.ndarray, dict[str, float], bool]:
    """One :class:`WBCPolicy` tick on a fresh policy.

    Returns ``(fed observation, action dict, main_session_ran)``. Exactly one of
    the two sessions runs per tick, so the flag is the whole session-choice
    channel and the fed array is read from whichever ran.
    """
    p = _make_policy(walk=True)
    actions = p.get_actions_sync(_full_g1_obs(), instruction, **goal)
    assert len(actions) == 1, "WBC is a per-tick controller: exactly one action dict"

    main_calls, walk_calls = p.policy_session.calls, p.walk_session.calls
    assert bool(main_calls) != bool(walk_calls), "exactly one session must run per tick"
    fed = (main_calls or walk_calls)[0]
    return np.asarray(fed), actions[0], bool(main_calls)


def _tick_gait(instruction: str, **goal: Any) -> tuple[np.ndarray, dict[str, float]]:
    """One :class:`WBCGaitPolicy` tick on a fresh policy (single session)."""
    p = WBCGaitPolicy(allow_missing_models=True)
    p.policy_session = _StubSession()
    p.set_robot_state_keys(_g1_keys())
    actions = p.get_actions_sync(_full_g1_obs(), instruction, **goal)
    assert len(actions) == 1
    return np.asarray(p.policy_session.calls[0]), actions[0]


class TestWBCPolicyDiscardsTheInstruction:
    """A goal stated in words changes neither the observation nor the action."""

    @pytest.mark.parametrize("instruction", _GOAL_BEARING_INSTRUCTIONS)
    def test_neither_the_observation_nor_the_action_changes(self, instruction: str) -> None:
        base_fed, base_actions, base_main = _tick_wbc(_BASELINE_INSTRUCTION, **_GOAL_KWARGS)
        fed, actions, main = _tick_wbc(instruction, **_GOAL_KWARGS)

        # Session choice first: a coarser regression than a shifted command
        # value, and it reports as a bool rather than a 570-float array diff.
        assert main is base_main is False, "a non-zero velocity must run the walk session"
        np.testing.assert_array_equal(fed, base_fed, err_msg=f"instruction {instruction!r} reached the observation")
        assert actions == base_actions, f"instruction {instruction!r} reached the action"

    @pytest.mark.parametrize("instruction", _GOAL_BEARING_INSTRUCTIONS)
    def test_a_non_empty_instruction_is_accepted_with_no_goal_kwargs(self, instruction: str) -> None:
        """The hardware/mesh shape: instruction only, no kwargs (see module docstring).

        Deliberately not a refusal - the uniform ``Policy`` interface takes an
        instruction on every provider. It must be accepted, ignored, and still
        yield the config-default goal's action.
        """
        fed, actions, main = _tick_wbc(instruction)
        base_fed, base_actions, base_main = _tick_wbc(_BASELINE_INSTRUCTION)

        assert main is base_main is True, "a zero default velocity must run the main session"
        np.testing.assert_array_equal(fed, base_fed)
        assert actions == base_actions
        assert len(actions) == 15 and all(np.isfinite(v) for v in actions.values())


class TestWBCGaitPolicyDiscardsTheInstruction:
    """Same contract on the gait variant, whose command block is 8 wide and
    whose observation carries a phase-clock tail derived from it - so a steered
    velocity or step frequency would show up in the fed array too."""

    @pytest.mark.parametrize("instruction", _GOAL_BEARING_INSTRUCTIONS)
    def test_neither_the_observation_nor_the_action_changes(self, instruction: str) -> None:
        goal = {**_GOAL_KWARGS, "gait_frequency": 1.5}
        base_fed, base_actions = _tick_gait(_BASELINE_INSTRUCTION, **goal)
        fed, actions = _tick_gait(instruction, **goal)

        np.testing.assert_array_equal(fed, base_fed, err_msg=f"instruction {instruction!r} reached the observation")
        assert actions == base_actions, f"instruction {instruction!r} reached the action"

    @pytest.mark.parametrize("instruction", _GOAL_BEARING_INSTRUCTIONS)
    def test_a_non_empty_instruction_is_accepted_with_no_goal_kwargs(self, instruction: str) -> None:
        fed, actions = _tick_gait(instruction)
        base_fed, base_actions = _tick_gait(_BASELINE_INSTRUCTION)
        np.testing.assert_array_equal(fed, base_fed)
        assert actions == base_actions


class TestTheMeasuredChannelsAreLive:
    """The two channels the pins above assert are UNCHANGED must be channels
    that can change, or the pins pass by measuring a constant.

    Each case supplies a probe value through the documented kwarg spelling and
    asserts the same comparison the pins make comes out different.
    """

    @pytest.mark.parametrize(
        "override",
        [
            pytest.param({"target_velocity": [1.0, 0.0, 0.0]}, id="velocity"),
            pytest.param({"target_orientation": [0.4, 0.0, 0.0]}, id="orientation"),
            pytest.param({"height": 0.8}, id="height"),
        ],
    )
    def test_the_fed_observation_moves_for_a_goal_supplied_the_documented_way(self, override: dict[str, Any]) -> None:
        base_fed, _, _ = _tick_wbc(_BASELINE_INSTRUCTION, **_GOAL_KWARGS)
        fed, _, _ = _tick_wbc(_BASELINE_INSTRUCTION, **{**_GOAL_KWARGS, **override})
        assert not np.array_equal(fed, base_fed), f"{override} did not reach the observation"

    def test_the_session_choice_moves_for_a_zero_velocity(self) -> None:
        _, _, walking_main = _tick_wbc(_BASELINE_INSTRUCTION, **_GOAL_KWARGS)
        _, _, standing_main = _tick_wbc(_BASELINE_INSTRUCTION, **{**_GOAL_KWARGS, "target_velocity": [0.0, 0.0, 0.0]})
        assert walking_main is False and standing_main is True

    def test_the_gait_fed_observation_moves_for_a_step_frequency(self) -> None:
        goal = {**_GOAL_KWARGS, "gait_frequency": 1.5}
        base_fed, _ = _tick_gait(_BASELINE_INSTRUCTION, **goal)
        fed, _ = _tick_gait(_BASELINE_INSTRUCTION, **{**goal, "gait_frequency": 2.5})
        assert not np.array_equal(fed, base_fed), "gait_frequency did not reach the observation"

    def test_the_probe_strings_are_not_all_inert_text(self) -> None:
        """Two probes name a goal in a spelling something in this tree parses.

        Guards the probe SET rather than the policy: if these were reduced to
        prose the pins would still pass, but they would stop covering the
        realistic regression (copying curobo's JSON-in-instruction fallback, or
        adding a ``key=value`` parse) that motivates them.
        """
        json_probes = [i for i in _GOAL_BEARING_INSTRUCTIONS if i.startswith("{")]
        assert json_probes, "no JSON-shaped probe left in the set"
        for probe in json_probes:
            assert "target_velocity" in json.loads(probe)

        assert any(
            "=" in i and i.split("=", 1)[0] in {"target_orientation", "target_velocity", "height"}
            for i in _GOAL_BEARING_INSTRUCTIONS
        ), "no key=value-shaped probe left in the set"
