# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An RL example that says the trained actor is rolled out contains that rollout.

Measured on ``c68444952``: both ``examples/training/train_ppo_reach.py`` and
``examples/training/train_fastsac_reach.py`` closed their module docstring with
"the deterministic (mean) policy is rolled out ... and the joint trajectory is
reported", and neither file contained a rollout call. ``main()`` stopped at::

    final metrics={'mean_reward': -0.0275, 'mean_episode_return': -1.3773, ...}

A reader who ran the file to watch the trained actor move got four metric lines
and no trajectory, and ``mean_reward`` does not say where the joint ended up -
the promise the docstring makes is the only readout that does. The deploy half
the promise names is a real path (``docs/training/rl.md`` "Deploying the
checkpoint"), so the fix was to perform the claim rather than delete it.

The rule is the pair, both halves keyed on the file rather than on this module's
memory of it: the docstring still carries the promise, and the code still
carries a rollout of ``policy_provider="rl"`` on the robot ``make_env`` builds.
The second half is the same constraint
``tests/test_docs_rl_deploy_fence_names_the_trained_robot.py`` puts on the docs
fences - ``RLCheckpointPolicy`` binds ``actor_obs_keys`` by name, so a
checkpoint trained on the SO-100's ``Elbow`` cannot be rolled out on a robot
whose joints are ``1``..``6``.

The rollout's verdict is the third half of the pair. ``run_policy`` does not
raise - every failure comes back as a ``{"status": "error", ...}`` envelope, a
policy that will not construct included - so an example that discards the return
and indexes its trace turns every rollout failure into ``IndexError: list index
out of range`` with the runner's message gone. :func:`_status_is_read` pins that
the return is bound to a name and that name's ``["status"]`` is compared.

An example that stops promising a rollout has to leave :data:`_PROMISE_MAKERS`
in the same change, which is what keeps the rewording from silencing the rule;
:func:`test_no_other_example_promises_a_rollout_unwatched` is the other
direction, so a new RL example inherits the pair.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

#: Examples whose docstring sells the trained actor being rolled out.
_PROMISE_MAKERS = (
    "examples/training/train_ppo_reach.py",
    "examples/training/train_fastsac_reach.py",
)

#: The phrase that makes the promise, in the spelling both examples use.
_PROMISE = "is rolled out"

#: Calls that reach a trained checkpoint's actor: the rollout paths
#: ``docs/training/rl.md`` documents for a ``TrainResult.checkpoint_dir``.
_ROLLOUT_CALLS = frozenset({"run_policy", "eval_policy", "load_deployable_actor"})


def _module(path: str) -> tuple[str, ast.Module]:
    source = (_REPO_ROOT / path).read_text(encoding="utf-8")
    return source, ast.parse(source, filename=path)


def _called_name(node: ast.Call) -> str | None:
    """The bare name of a call target, through an attribute access if needed."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def _calls(tree: ast.Module, names: frozenset[str]) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _called_name(n) in names]


def _string_keyword(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def _first_string_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _bound_to(tree: ast.Module, call: ast.Call) -> str | None:
    """The name a call's return is assigned to, or ``None`` when it is discarded."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.value is call:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                return target.id
    return None


def _status_is_read(tree: ast.Module, name: str) -> bool:
    """``name["status"]`` appears as one side of a comparison."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for side in (node.left, *node.comparators):
            if (
                isinstance(side, ast.Subscript)
                and isinstance(side.value, ast.Name)
                and side.value.id == name
                and isinstance(side.slice, ast.Constant)
                and side.slice.value == "status"
            ):
                return True
    return False


def _trained_robot(tree: ast.Module) -> str:
    """The robot ``make_env`` constructs - the only one the checkpoint fits."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "make_env":
            for call in _calls(ast.Module(body=node.body, type_ignores=[]), frozenset({"Robot"})):
                named = _first_string_arg(call)
                if named:
                    return named
    raise AssertionError("the example defines make_env building Robot('<name>')")


@pytest.mark.parametrize("path", _PROMISE_MAKERS)
def test_the_example_rolls_out_the_checkpoint_it_promises(path: str) -> None:
    """The docstring's promise and the rollout that keeps it, in one file."""
    source, tree = _module(path)
    docstring = ast.get_docstring(tree) or ""

    assert _PROMISE in docstring, (
        f"{path} no longer promises a rollout; drop it from _PROMISE_MAKERS in the same change"
    )
    rollouts = _calls(tree, _ROLLOUT_CALLS)
    assert rollouts, f"{path} promises '{_PROMISE}' and calls none of {sorted(_ROLLOUT_CALLS)}"

    trained = _trained_robot(tree)
    for call in rollouts:
        if _called_name(call) != "run_policy":
            continue
        assert _string_keyword(call, "policy_provider") == "rl", (
            f"{path} rolls the checkpoint out through a provider other than 'rl'"
        )
        assert _string_keyword(call, "robot_name") == trained, (
            f"{path} rolls the {trained}-trained checkpoint out on another robot; "
            "RLCheckpointPolicy binds actor_obs_keys by name and refuses an observation missing one"
        )
        bound = _bound_to(tree, call)
        assert bound is not None, (
            f"{path} discards run_policy's return; it does not raise, so a failed rollout "
            "is only visible in the status envelope it hands back"
        )
        assert _status_is_read(tree, bound), (
            f'{path} never compares {bound}["status"]; a rollout that fails leaves an empty '
            "trace and the example would surface an IndexError instead of the runner's message"
        )
    assert f'Robot("{trained}"' in source, f"{path} constructs the robot it rolls the checkpoint out on"


def test_no_other_example_promises_a_rollout_unwatched() -> None:
    """A new example making the promise joins the table rather than escaping it."""
    promising = sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _EXAMPLES_DIR.rglob("*.py")
        if _PROMISE in (ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "")
    )

    assert promising == sorted(_PROMISE_MAKERS), promising
