"""A mesh command key the peer does not read is refused by name, not dropped.

``validate_command`` built its output only from keys it validated and dropped
the rest, so an unknown key was never forwarded - but never reported either. A
typo therefore handed control to the default the typo was meant to override:
two local peers, ``send(peer, {"action": "execute", "instruction": "wave",
"policy_provider": "mock", "durration": 0.5})`` answered ``success`` after
``30.0s | 1500 steps`` (the 30 s default), and ``"n_step": 5`` did the same. In
process, ``run_policy(durration=0.5)`` is refused with the valid names; the wire
was the one surface that guessed. On hardware that is a robot asked to move for
half a second moving for thirty.

Pinned here: an unread key is refused naming it and the closest key the action
does read; routing fields and every key the validator reads still pass; the
refusal comes back from a real peer before anything moves
(``tests_integ/mesh/test_a_misspelled_command_key_moves_nothing.py``); and :data:`COMMAND_KEYS`, which words the
refusal, names exactly the keys the validator's source reads.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from strands_robots.mesh import security
from strands_robots.mesh.security import COMMAND_KEYS, ValidationError, validate_command

_EXECUTE = {"action": "execute", "instruction": "wave", "policy_provider": "mock"}


@pytest.mark.parametrize(
    ("typo", "meant"),
    [
        ("durration", "duration"),
        ("n_step", "n_steps"),
        ("control_freq", "control_frequency"),
        ("robotname", "robot_name"),
    ],
)
def test_a_misspelled_execute_key_is_refused_with_the_key_it_meant(typo, meant):
    with pytest.raises(ValidationError) as caught:
        validate_command({**_EXECUTE, typo: 1})

    text = str(caught.value)
    assert text.startswith(f"execute: unknown key(s) {typo!r} (did you mean {meant!r}?)"), text
    assert "defaults would apply" in text


def test_a_key_another_action_reads_is_refused_on_this_one():
    with pytest.raises(ValidationError, match=r"status: unknown key\(s\) 'duration'.*status takes no keys"):
        validate_command({"action": "status", "duration": 3})


def test_routing_fields_and_every_read_key_still_pass():
    out = validate_command({**_EXECUTE, "duration": 0.5, "n_steps": 5, "turn_id": "t-1", "sender_id": "op"})

    assert out["duration"] == 0.5 and out["n_steps"] == 5 and out["turn_id"] == "t-1"


def _keys_read_by(function) -> set[str]:
    """String keys a function reads off its ``cmd`` argument (``.get``, ``in``, ``[]``)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cmd"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        elif isinstance(node, ast.Compare) and any(isinstance(op, ast.In) for op in node.ops):
            if (
                isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
                and any(isinstance(c, ast.Name) and c.id == "cmd" for c in node.comparators)
            ):
                keys.add(node.left.value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cmd"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def test_the_key_table_names_exactly_what_the_validator_reads():
    """Drift here only degrades the hint - the refusal reads the validator's output - but a
    hint that omits a key the peer reads would send an operator looking for the wrong name."""
    read = _keys_read_by(security.validate_command) | _keys_read_by(security._validate_sim_call)
    read |= _keys_read_by(security._validate_call)
    passthrough = {"turn_id", "sender_id"}  # read through a loop variable, every action
    tabled = set().union(*COMMAND_KEYS.values())

    assert read - {"action"} == tabled, (sorted(read - {"action"} - tabled), sorted(tabled - read))
    assert passthrough & tabled == set()


def test_a_per_robot_stop_travels_because_the_peer_reads_its_robot_name():
    """``stop`` is the one action where a refusal is fail-open: the peer's dispatcher reads
    ``stop.robot_name`` (per-robot ``stop_policy``), so the key must reach it, coerced like
    ``set_joints.robot_name`` is, and a stop with no robot name still passes untouched."""
    out = validate_command({"action": "stop", "robot_name": "so101", "turn_id": "t-1"})

    assert out["robot_name"] == "so101" and out["turn_id"] == "t-1"
    assert validate_command({"action": "stop"}) == {"action": "stop"}
    assert "robot_name" in COMMAND_KEYS["stop"]
    with pytest.raises(ValidationError):
        validate_command({"action": "stop", "robot_name": "../so101"})
