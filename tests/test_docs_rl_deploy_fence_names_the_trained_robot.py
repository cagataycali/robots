"""The RL deploy fences must roll the checkpoint out on the robot it was trained on.

``RLCheckpointPolicy`` binds ``actor_obs_keys`` by name and refuses an
observation that omits one (``"observation omits actor_obs_keys the ppo
checkpoint was trained on"``). ``docs/training/rl.md`` trains through a
``make_env`` that builds ``Robot("so100")``, whose joints are ``Elbow`` and
friends; the SO-101's are ``1``..``6``. So the ``run_policy`` fence on that page
and its twin on ``docs/policies/rl.md`` can only run on the robot ``make_env``
constructed - any other name is refused before the first action.

This reads the robot ``make_env`` constructs off the training page and grades
every ``policy_provider="rl"`` deploy fence on both pages against it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_TRAINING_PAGE = _REPO_ROOT / "docs" / "training" / "rl.md"
_DEPLOY_PAGES = (_TRAINING_PAGE, _REPO_ROOT / "docs" / "policies" / "rl.md")
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _parsed_fences(page: Path) -> list[ast.Module]:
    trees: list[ast.Module] = []
    for fence in _PYTHON_FENCE.findall(page.read_text(encoding="utf-8")):
        try:
            trees.append(ast.parse(fence))
        except SyntaxError:
            continue  # prose placeholders (``...`` after a keyword) are not fences under test
    return trees


def _is_robot_call(node: ast.Call) -> bool:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == "Robot"


def _robot_name(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def _keyword(node: ast.Call, name: str) -> str | None:
    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _trained_robot() -> str:
    for tree in _parsed_fences(_TRAINING_PAGE):
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == "make_env":
                for call in ast.walk(fn):
                    if isinstance(call, ast.Call) and _is_robot_call(call) and _robot_name(call):
                        return _robot_name(call)  # type: ignore[return-value]
    raise AssertionError("docs/training/rl.md defines make_env building Robot('<name>')")


def _rl_deploy_fences(page: Path) -> list[ast.Module]:
    fences = []
    for tree in _parsed_fences(page):
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "run_policy"
                and _keyword(call, "policy_provider") == "rl"
            ):
                fences.append(tree)
                break
    return fences


@pytest.mark.parametrize("page", _DEPLOY_PAGES, ids=lambda p: p.relative_to(_REPO_ROOT).as_posix())
def test_rl_deploy_fence_names_the_robot_make_env_trained_on(page: Path) -> None:
    trained = _trained_robot()
    fences = _rl_deploy_fences(page)
    assert fences, f"{page.name} carries a run_policy(policy_provider='rl') fence"

    for tree in fences:
        named: dict[str, str | None] = {}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            if _is_robot_call(call):
                named["Robot(...)"] = _robot_name(call)
            elif isinstance(call.func, ast.Attribute) and call.func.attr == "run_policy":
                named["run_policy(robot_name=...)"] = _keyword(call, "robot_name")
        wrong = {spelling: name for spelling, name in named.items() if name != trained}
        assert not wrong, (
            f"deploy fence rolls the {trained}-trained checkpoint out on another robot: {wrong}; "
            f"RLCheckpointPolicy refuses an observation that omits the trained actor_obs_keys"
        )
        assert "Robot(...)" in named, "the deploy fence constructs the Robot it calls run_policy on"
