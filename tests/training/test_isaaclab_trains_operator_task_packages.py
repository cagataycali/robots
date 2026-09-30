"""An operator's own Isaac Lab task package is trained, played and validated like Isaac Lab's own.

Isaac Lab trains a task from an outside package when ``--external_callback
<module>.<function>`` registers it. strands could not: ``validate()`` scanned only
``isaaclab_tasks`` (so "Acme-Cartpole-Short-v0 is not registered ... did you mean
['Isaac-Cartpole-Direct']"), ``extra`` had no way to name the callback, and the
child environment drops ``PYTHONPATH`` on purpose. The operator now names the
packages in ``$STRANDS_ISAACLAB_TASK_PACKAGES`` (``module:function``, importable in
the Isaac Lab venv), and the task gets the callback on train and play.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from strands_robots.training import _isaaclab_runtime as runtime
from tests.training.test_isaaclab import _poll, _spec, _trainer, fake_python  # noqa: F401

_ACME = textwrap.dedent(
    """\
    import gymnasium as gym

    def register_tasks():
        gym.register(id="Acme-Cartpole-Short-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv")
    """
)


@pytest.fixture
def venv_with_packages(fake_python: Path) -> Path:  # noqa: F811
    """Isaac Lab's own tasks plus an operator package, in the fake interpreter's venv."""
    site = fake_python.parent.parent / "lib" / "python3.12" / "site-packages"
    (site / "isaaclab_tasks").mkdir(parents=True)
    (site / "isaaclab_tasks" / "__init__.py").write_text(
        'import gymnasium as gym\\ngym.register(id="Isaac-Cartpole")\\n'
    )
    (site / "acme_tasks").mkdir()
    (site / "acme_tasks" / "__init__.py").write_text(_ACME)
    runtime._TASK_CACHE.clear()
    return site


def test_an_unnamed_package_is_not_trained_and_the_refusal_says_how(
    venv_with_packages: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(runtime.TASK_PACKAGES_ENV, raising=False)
    problems = _trainer().validate(_spec(tmp_path, task="Acme-Cartpole-Short-v0"))
    assert any("is not registered" in p and runtime.TASK_PACKAGES_ENV in p for p in problems), problems


def test_a_named_package_trains_through_the_external_callback(
    venv_with_packages: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(runtime.TASK_PACKAGES_ENV, "acme_tasks:register_tasks")
    runtime._TASK_CACHE.clear()
    trainer = _trainer()
    spec = _spec(tmp_path, task="Acme-Cartpole-Short-v0")
    assert trainer.validate(spec) == []
    cmd = trainer.build_command(spec, "j")
    assert cmd[cmd.index("--external_callback") + 1] == "acme_tasks.register_tasks"
    job = _poll(trainer, trainer.train(spec).job_id)
    assert job.status == "success", job.message
    played = trainer.play(job.job_id, video_length=5)
    assert played.status != "error", played.message
    _poll(trainer, played.job_id)
    argv = json.loads(Path(str(tmp_path / "argv.json") + ".play").read_text())
    assert argv[argv.index("--external_callback") + 1] == "acme_tasks.register_tasks"


def test_isaac_labs_own_tasks_get_no_callback(
    venv_with_packages: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(runtime.TASK_PACKAGES_ENV, "acme_tasks:register_tasks")
    assert "--external_callback" not in _trainer().build_command(_spec(tmp_path), "j")


def test_a_package_named_without_its_function_is_refused(
    venv_with_packages: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(runtime.TASK_PACKAGES_ENV, "acme_tasks")
    runtime._TASK_CACHE.clear()
    problems = _trainer().validate(_spec(tmp_path, task="Acme-Cartpole-Short-v0"))
    assert any("'acme_tasks:<function>'" in p for p in problems), problems


def test_malformed_entries_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.TASK_PACKAGES_ENV, "acme_tasks:register_tasks, ;rm -rf /, pkg.sub:fn,9bad")
    assert runtime.operator_task_packages() == {"acme_tasks": "register_tasks", "pkg.sub": "fn"}
