# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""From-scratch RL is an extra of its own, and an install without it is told so.

``strands_robots.training.rl`` computes in torch - ``env.py``,
``normalization.py``, ``replay_buffer.py`` and ``vec_env.py`` import it at
their top - and ``GymSimEnv`` presents a ``SimEnv`` through gymnasium. Neither
package was declared by an extra for this purpose: torch arrived only inside
``[kimodo]`` / ``[lerobot]`` / ``[cosmos3-diffusers]`` / ``[sim-gs]`` and
gymnasium only inside ``lerobot``. ``docs/training/rl.md`` had no install line,
``gym_env.py`` said "the ``[sim]`` extra pulls it in" (it does not: ``[sim]`` is
``robot_descriptions`` alone), and the three trainers' own
``require_optional("torch", ...)`` calls sat inside ``setup()``, behind a
package ``__init__`` that had already imported torch bare. Measured on a fresh
``pip install strands-robots`` (main ``6abbd1a25``), the docs page's first
line::

    >>> from strands_robots.training import create_trainer
    >>> create_trainer("ppo")
    ModuleNotFoundError: No module named 'torch'

Three things change together, and the cells below pin each: the manifest gains
``rl = [strands-robots[sim-mujoco], torch, gymnasium]`` folded into ``[all]``;
the four modules that imported torch bare (``env``, ``normalization``,
``replay_buffer``, ``vec_env``) bind it through ``require_optional(...,
extra="rl")`` under a ``TYPE_CHECKING`` split that keeps mypy's view intact, so
every door into the package - the package itself, a submodule,
``create_trainer`` - refuses with the extra's name; and the sites that already
used ``require_optional`` name the extra too.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import strands_robots
from tests._blocked_module import blocked
from tests._module_reimport import reimport

_PYPROJECT = Path(strands_robots.__file__).resolve().parent.parent / "pyproject.toml"
_RL_DIR = Path(strands_robots.__file__).resolve().parent / "training" / "rl"
#: The modules that bind torch at import. Their cached copies would answer a
#: re-import of the package from the memo, so a cell that grades the gate has to
#: run each of their top levels again.
_TORCH_MODULES = ("env", "normalization", "replay_buffer", "vec_env")


def _forget_rl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every torch-binding module's top level again under the block, then the package's."""
    for leaf in _TORCH_MODULES:
        with pytest.raises(ImportError):
            reimport(monkeypatch, f"strands_robots.training.rl.{leaf}")


def _extras() -> dict[str, list[Requirement]]:
    manifest = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return {
        name: [Requirement(spec) for spec in specs]
        for name, specs in manifest["project"]["optional-dependencies"].items()
    }


def test_the_rl_extra_declares_torch_gymnasium_and_the_mujoco_backend() -> None:
    """``[rl]`` carries what the package imports, plus the SimEngine the env adapters step."""
    extras = _extras()
    assert "rl" in extras, sorted(extras)
    by_name = {req.name: req for req in extras["rl"]}
    assert "torch" in by_name, by_name
    assert "gymnasium" in by_name, by_name
    # ``strands-robots[sim-mujoco]``: the env adapters step a MuJoCo SimEngine.
    self_ref = by_name.get("strands-robots")
    assert self_ref is not None, by_name
    assert "sim-mujoco" in self_ref.extras, self_ref


def test_the_all_bundle_folds_the_rl_extra_in() -> None:
    """``[all]`` is the bundle the docs and hatch env install; a new extra must join it."""
    all_extras = {extra for req in _extras()["all"] if req.name == "strands-robots" for extra in req.extras}
    assert "rl" in all_extras, sorted(all_extras)


def test_importing_the_rl_package_without_torch_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is in the package ``__init__``, before any submodule can import torch bare."""
    with blocked("torch"):
        _forget_rl(monkeypatch)
        with pytest.raises(ImportError) as info:
            reimport(monkeypatch, "strands_robots.training.rl")

    text = str(info.value)
    assert info.value.name == "torch", text
    assert "pip install 'strands-robots[rl]'" in text, text
    assert "strands_robots.training.rl" in text, text


def test_create_trainer_ppo_without_torch_names_the_extra() -> None:
    """The docs page's first line, ``create_trainer("ppo")``, gets the same refusal.

    Run in a fresh interpreter: the factory's loader is ``from
    strands_robots.training.rl.ppo import PpoTrainer``, and ``import a.b.c``
    answers from ``sys.modules["a.b.c"]`` without touching the parent, so in
    this session a cached ``ppo`` would let the loader succeed and grade the
    wrong thing. A core install has no cache; neither does the subprocess.
    """
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from strands_robots.training import create_trainer\n"
        "try:\n"
        "    create_trainer('ppo')\n"
        "except ImportError as exc:\n"
        "    print(exc.name); print(exc); raise SystemExit(0)\n"
        "raise SystemExit('create_trainer succeeded without torch')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=_PYPROJECT.parent
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("torch\n"), done.stdout
    assert "pip install 'strands-robots[rl]'" in done.stdout, done.stdout


def test_the_existing_require_optional_sites_name_the_extra() -> None:
    """Every torch/gymnasium site says ``[rl]``, and no module imports torch bare any more."""
    texts = {path.name: path.read_text(encoding="utf-8") for path in _RL_DIR.glob("*.py")}
    sites = {
        name: text
        for name, text in texts.items()
        if 'require_optional("torch"' in text or 'require_optional("gymnasium"' in text
    }
    expected = {f"{leaf}.py" for leaf in _TORCH_MODULES} | {"ppo.py", "fast_sac.py", "fast_td3.py", "gym_env.py"}
    assert expected <= set(sites), sorted(sites)
    bare_remedy = [
        name
        for name, text in sites.items()
        if 'require_optional("torch", purpose=' in text or 'require_optional("gymnasium", purpose=' in text
    ]
    assert not bare_remedy, bare_remedy
    bare_import = [
        f"{name}:{lineno}"
        for name, text in texts.items()
        for lineno, line in enumerate(text.splitlines(), 1)
        if line in ("import torch", "from torch import nn") or line.startswith("from torch ")
    ]
    assert not bare_import, bare_import
    assert "``[sim]`` extra" not in texts["gym_env.py"]
