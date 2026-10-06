"""``close_match_hint`` suppresses a suggestion that shares no semantic meaning.

:func:`strands_robots.simulation.base.close_match_hint` is the shared helper
behind every unknown-entity refusal on the agent-facing MuJoCo sim surface
(unknown action / model / object / camera / robot / parameter), so its
cutoff is in every one of those messages at once.

Before this pin the cutoff was ``0.4`` - a lone outlier in the codebase (every
other difflib call in strands_robots is 0.5-0.8, and the same-module
``_MISSPELLING_RATIO`` is 0.8). ``0.4`` scored character-overlap matches that
carry no semantic meaning: the README's hero prompt "pick up the red cube"
naturally produces ``action='pick'`` from an LLM, whose refusal returned
``Did you mean: run_policy, stop_policy, eval_policy`` with equal weight and
sent a caller taking the first suggestion into ``stop_policy`` (kills the
current controller) rather than toward recovery.

The one-edit typo pin at
``tests/simulation/mujoco/test_unknown_action_message_suggests_a_published_action.py``
keeps covering the recovery case (``renderr`` -> ``render``,
``set_joint_position`` -> ``set_joint_positions``, ``get_stat`` -> ``get_state``);
those all score well above the stdlib default ``0.6`` and are unaffected.
This file pins the other half: a semantically unrelated input receives no
suggestion, so the caller is not nudged onto a destructive sibling.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import close_match_hint


# A fixed vocabulary that reproduces the sim's published action enum without
# importing it (keeps this pin independent of enum churn).
_ACTIONS = [
    "add_camera",
    "add_object",
    "add_robot",
    "apply_force",
    "destroy",
    "eval_policy",
    "get_body_state",
    "get_contacts",
    "get_robot_state",
    "get_state",
    "list_cameras",
    "list_objects",
    "list_robots",
    "load_scene",
    "move_object",
    "move_to",
    "randomize",
    "raycast",
    "render",
    "render_depth",
    "replace_scene_mjcf",
    "reset",
    "run_policy",
    "save_state",
    "set_geom_properties",
    "set_gravity",
    "set_joint_positions",
    "set_obs_noise",
    "set_timestep",
    "start_policy",
    "start_recording",
    "step",
    "stop_policy",
    "stop_recording",
]


# Inputs whose prior-art 0.4 suggestions were character-overlap noise:
# ``pick`` -> run_policy/stop_policy/eval_policy; ``grab`` -> set_gravity;
# ``grasp`` -> raycast/step/reset; ``help`` -> step/eval_policy;
# ``place`` -> load_scene/apply_force; ``spawn`` -> step.
_SEMANTIC_MISSES = ["pick", "grab", "grasp", "help", "place", "spawn", "put"]


@pytest.mark.parametrize("name", _SEMANTIC_MISSES)
def test_a_semantic_miss_receives_no_suggestion(name: str) -> None:
    """A short name sharing no root with any action produces no suggestion.

    The fragment this helper returns is appended to the verdict, so the
    caller still sees the pointer at ``tool_spec`` the enclosing message
    carries. The suppression is purely of the misleading ``Did you mean:``
    half - recovery is preserved.
    """
    hint = close_match_hint(name, _ACTIONS)
    assert hint == "", (
        f"close_match_hint({name!r}, ...) returned {hint!r}; a semantic miss "
        "must not surface a character-overlap suggestion (that was the 0.4 "
        "cutoff footgun)."
    )


# One-edit typos still receive the recovery suggestion they always have.
# These sit above difflib's stdlib default 0.6 cutoff.
_ONE_EDIT = [
    ("renderr", "render"),
    ("add_objec", "add_object"),
    ("list_robot", "list_robots"),
    ("get_stat", "get_state"),
    ("set_joint_position", "set_joint_positions"),
]


@pytest.mark.parametrize(("typo", "want"), _ONE_EDIT)
def test_a_one_edit_typo_still_suggests_the_name(typo: str, want: str) -> None:
    """The pin on the recovery case: a near-miss still names the right action.

    Without this a cutoff raised too far would silence the suggestion the
    unknown-action test suite covers, breaking the one-edit recovery that
    is the primary purpose of this helper.
    """
    hint = close_match_hint(typo, _ACTIONS)
    assert want in hint, (
        f"close_match_hint({typo!r}, ...) returned {hint!r}; a one-edit typo "
        f"of {want!r} must still receive it as a suggestion."
    )
