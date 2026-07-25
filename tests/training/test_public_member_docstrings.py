# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The training package public API must document every public member.

The :mod:`strands_robots.training` package is the training-provider surface an
agent drives to fine-tune or RL-train a policy: the
:class:`~strands_robots.training.base.Trainer` ABC and its dataclasses
(:class:`~strands_robots.training.base.TrainSpec` /
:class:`~strands_robots.training.base.TrainResult`), the provider factory in
:mod:`~strands_robots.training.factory`
(:func:`~strands_robots.training.factory.create_trainer` and friends), the
reward-model helpers in :mod:`~strands_robots.training.reward`, and the
reinforcement-learning stack in :mod:`~strands_robots.training.rl`
(:class:`~strands_robots.training.rl.ppo.PpoTrainer`,
:class:`~strands_robots.training.rl.fast_sac.FastSacTrainer`, the vectorised
:class:`~strands_robots.training.rl.vec_env.VecSimEnv`, and the shared
:class:`~strands_robots.training.rl.base_algo.BaseRLAlgo`). Agents and
integrators read these docstrings to drive the surface, so each public class,
method, property, and module-level function must state its own behavior.

A finer-grained sibling guard
(``tests/training/test_trainer_provider_docstrings.py``) already pins the four
concrete ``Trainer`` subclasses, but the factory, reward helpers, dataclasses,
and the whole ``rl`` subpackage were left unguarded. This guard closes that gap
package-wide, matching the peer guards under ``tests/*/`` (e.g. the inference,
tools, and policies guards).

The scan walks the package modules by AST (no import, so it never needs the
optional ``training`` / ``rl`` extras installed) and fails if any public class,
public method/property, or public module-level function lacks a docstring. It
also pins the discovered public surface so a refactor that drops or renames a
class/function trips the guard instead of silently shrinking the scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.training as training_pkg

_PACKAGE_DIR = Path(training_pkg.__file__).parent

# Public-API modules, keyed by their path relative to the package dir. Private
# modules (``_inproc`` / ``_validate``) and the re-export-only ``__init__`` are
# out of scope. The ``rl`` subpackage is scanned alongside the top level.
_MODULES = (
    "base.py",
    "cosmos3.py",
    "factory.py",
    "groot.py",
    "lerobot.py",
    "mock.py",
    "reward.py",
    "rl/base_algo.py",
    "rl/env.py",
    "rl/fast_sac.py",
    "rl/gym_env.py",
    "rl/normalization.py",
    "rl/ppo.py",
    "rl/replay_buffer.py",
    "rl/vec_env.py",
)

# Every public class the package exposes, keyed ``module.py::ClassName``. Pinned
# so a refactor that drops or renames a class trips the completeness guard
# instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "base.py::TrainSpec",
    "base.py::TrainResult",
    "base.py::Trainer",
    "cosmos3.py::Cosmos3Trainer",
    "groot.py::Gr00tTrainer",
    "lerobot.py::LerobotTrainer",
    "mock.py::MockTrainer",
    "rl/base_algo.py::RLTrainSpec",
    "rl/base_algo.py::BaseRLAlgo",
    "rl/env.py::SimEnv",
    "rl/fast_sac.py::FastSacTrainer",
    "rl/normalization.py::EmpiricalNormalization",
    "rl/ppo.py::PpoTrainer",
    "rl/replay_buffer.py::SimpleReplayBuffer",
    "rl/vec_env.py::VecSimEnv",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "factory.py::register_trainer",
    "factory.py::list_trainers",
    "factory.py::import_trainer_class",
    "factory.py::create_trainer",
    "reward.py::compute_rabc_weights",
    "reward.py::load_reward_model",
    "reward.py::reward_progress",
    "rl/gym_env.py::GymSimEnv",
    "rl/ppo.py::compute_gae",
}


def _module_tree(module: str) -> ast.Module:
    """Parse one package module into an AST (no import)."""
    source_file = _PACKAGE_DIR / module
    return ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder methods (``__init__`` and friends) are out of scope: their contract
    is documented on the class docstring itself.
    """
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every public class in the modules."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def _public_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``module.py::func`` -> FunctionDef for every public module-level function."""
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                funcs[f"{module}::{node.name}"] = node
    return funcs


def test_modules_define_expected_public_surface() -> None:
    """Guard: the scan actually found the classes and functions it protects."""
    assert set(_public_classes()) == _EXPECTED_CLASSES, set(_public_classes())
    assert set(_public_functions()) == _EXPECTED_FUNCTIONS, set(_public_functions())


def test_public_classes_and_members_have_docstrings() -> None:
    offenders: dict[str, list[str]] = {}
    for qualname, node in _public_classes().items():
        missing = _public_members_without_docstring(node)
        if ast.get_docstring(node) is None:
            missing = ["<class docstring>", *missing]
        if missing:
            offenders[qualname] = missing
    assert not offenders, (
        "Every public class in strands_robots.training -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior (concrete Trainer/algorithm overrides must not lean on the "
        "base ABC's text). Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.training must "
        "have a docstring. Undocumented functions: " + repr(offenders)
    )
