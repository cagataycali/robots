"""Every backend's ``start_recording`` resolves the dataset dir the same way.

``start_recording`` is reimplemented per backend, and each copy has to answer the
same question before it touches the disk: which directory does ``repo_id`` /
``root`` name? :func:`~strands_robots.dataset_recorder.resolve_dataset_dir` is
the one answer -- it is what ``DatasetRecorder.create`` itself resolves with, so
a backend that computes its own is deciding where a dataset lives while the
recorder writes somewhere else.

The Newton backend used to hand-roll the resolution. Its first two branches
matched the resolver and the third hard-coded ``~/.cache/huggingface/lerobot``,
so it ignored ``$HF_LEROBOT_HOME`` -- the override LeRobot honours and the only
way to relocate the dataset home. Measured on a scene whose home is relocated,
with ``repo_id="user/ds"`` and ``root=None``:

* ``last_dataset_root`` named the stale ``~/.cache`` path while the MuJoCo
  backend named the configured one, for byte-identical arguments;
* ``overwrite=True`` removed the dataset at the stale path -- one the call never
  addressed and which lives outside the configured home -- and left the
  addressed one for ``create()`` to remove;
* the resume probe missed an existing dataset in the configured home, so
  appending an episode dead-ended in ``FileExistsError`` telling the caller to
  use ``DatasetRecorder.resume()`` instead -- i.e. to bypass the method they
  called;
* ``last_dataset_root``, which ``stop_recording(bucket=...)`` syncs and
  ``verify_dataset_episodes`` reads once the recorder has been dropped, named a
  directory the session never wrote to.

The behavioural half drives the Newton engine through a hand-built ``SimWorld``
(the resolution runs before any solver call) with a stub recorder, so it needs
neither Newton nor lerobot installed. The structural half pins the property for
the two backends whose simulators cannot be driven here, and for any backend
added later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import strands_robots.dataset_recorder as dr
from strands_robots.dataset_recorder import resolve_dataset_dir
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_JOINTS = ["Rotation", "Pitch", "Elbow"]


def _engine() -> NewtonSimEngine:
    """A Newton engine bound to a hand-built world, without the Warp stack.

    Mirrors ``tests/simulation/newton/test_dataset_recording.py``: the recording
    lifecycle touches no physics, so the engine is built via ``__new__`` and
    given only the attributes ``start_recording`` reads.
    """
    world = SimWorld()
    world.robots["so100"] = SimRobot(
        name="so100", urdf_path="so100.xml", data_config="so100", joint_names=list(_JOINTS)
    )
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


class _StubRecorder:
    """Stands in for ``DatasetRecorder`` and records which path was taken.

    ``create`` and ``resume`` are the two outcomes the dataset-dir resolution
    selects between, so the calls are recorded rather than the objects.
    """

    calls: list[str] = []

    @classmethod
    def create(cls, **kwargs: object) -> object:
        cls.calls.append("create")
        return object()

    @classmethod
    def resume(cls, **kwargs: object) -> object:
        cls.calls.append("resume")
        return object()


@pytest.fixture
def relocated_home(monkeypatch, tmp_path):
    """Point the shared resolver's dataset home at ``tmp_path``.

    Patches ``_lerobot_home`` rather than the environment because lerobot reads
    ``HF_LEROBOT_HOME`` into a module constant at import time, so setting the
    variable here would not move an already-imported home -- and lerobot need not
    be installed at all. This is the same seam
    ``tests/simulation/mujoco/test_recording_paths.py`` pins the MuJoCo backend
    through.
    """
    home = tmp_path / "relocated" / "lerobot"
    monkeypatch.setattr(dr, "_lerobot_home", lambda: home)
    monkeypatch.setattr(dr, "lerobot_dataset_import_error", lambda: None)
    monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: True)
    _StubRecorder.calls = []
    monkeypatch.setattr(dr, "DatasetRecorder", _StubRecorder)
    return home


@pytest.fixture
def contained_user_home(monkeypatch, tmp_path):
    """Redirect ``Path.home()`` into ``tmp_path`` for the duration of a test.

    The hard-coded default the fix removes is spelled relative to the user's home
    directory, so a test that wants to observe what it touched has to move that
    home -- reading or writing the developer's real
    ``~/.cache/huggingface/lerobot`` is not an option. Patched on ``pathlib.Path``
    itself because the resolution under test calls ``Path.home()`` directly.
    """
    home = tmp_path / "user_home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _seed_dataset(directory: Path) -> None:
    """Make ``directory`` look like a real LeRobotDataset (it has ``meta/``)."""
    (directory / "meta").mkdir(parents=True)
    (directory / "meta" / "info.json").write_text("{}")


class TestNewtonResolvesTheConfiguredDatasetHome:
    """The resolved directory is the one the recorder will actually write to."""

    def test_a_namespaced_repo_id_resolves_under_the_configured_home(self, relocated_home):
        """The stashed root is the resolver's answer, not the ``~/.cache`` default.

        ``last_dataset_root`` is the only record of where a finished dataset
        lives once ``stop_recording`` drops the recorder, so a stale value sends
        ``stop_recording(bucket=...)`` and ``verify_dataset_episodes`` at a
        directory this session never wrote to.
        """
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(relocated_home / "user" / "ds")

    @pytest.mark.parametrize(
        "repo_id",
        ["user/ds", "bare_local", "./relative_ds", "/tmp/absolute_ds"],
    )
    def test_every_repo_id_shape_agrees_with_the_shared_resolver(self, relocated_home, repo_id):
        """Not just the relocated branch: no branch may disagree with the resolver.

        The hand-rolled copy matched on the path-shaped ``repo_id`` branches and
        diverged only on the home-relative one, which is exactly why reading the
        two side by side did not surface it.
        """
        engine = _engine()

        result = engine.start_recording(repo_id=repo_id, root=None, fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(resolve_dataset_dir(repo_id, None))

    def test_an_explicit_root_still_wins(self, relocated_home, tmp_path):
        """``root=`` is used verbatim, so relocating the home cannot override it."""
        explicit = tmp_path / "explicit"
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=str(explicit), fps=30)

        assert result["status"] == "success", result
        assert engine._world._backend_state["last_dataset_root"] == str(explicit)


class TestOverwriteRemovesOnlyTheAddressedDataset:
    """``overwrite=True`` deletes a dataset, so it must delete the right one."""

    def test_overwrite_clears_the_dataset_inside_the_configured_home(self, relocated_home):
        addressed = relocated_home / "user" / "ds"
        _seed_dataset(addressed)
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30, overwrite=True)

        assert result["status"] == "success", result
        assert not addressed.exists(), "the dataset the call addressed survived overwrite=True"

    def test_overwrite_leaves_an_identically_named_dataset_under_the_default_home(
        self, relocated_home, contained_user_home
    ):
        """The destructive half, and the reason this is a bug rather than a mismatch.

        A dataset at the same ``repo_id`` under the ``~/.cache`` default is not
        the one ``overwrite=True`` was asked to replace once the home has been
        moved. Removing it destroys recorded episodes at a path the call never
        named, under ``status="success"``.

        ``Path.home`` is contained for this one assertion because the pre-fix
        path is only reachable through it; asserting against the real
        ``~/.cache/huggingface/lerobot`` is the one thing this test must not do.
        """
        bystander = contained_user_home / ".cache" / "huggingface" / "lerobot" / "user" / "ds"
        _seed_dataset(bystander)
        _seed_dataset(relocated_home / "user" / "ds")
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30, overwrite=True)

        assert result["status"] == "success", result
        assert (bystander / "meta" / "info.json").exists(), (
            "overwrite=True removed a dataset under the default home, which this call did not address"
        )


class TestAnExistingDatasetIsResumedNotRecreated:
    """The resume probe reads the resolved dir, so it must read the right one."""

    def test_an_existing_dataset_in_the_configured_home_is_resumed(self, relocated_home):
        """Missing it is not a slower path but a dead end.

        ``create()`` refuses a directory holding a ``meta/`` with a
        ``FileExistsError`` naming ``overwrite=True`` (which discards the
        recorded episodes) and ``DatasetRecorder.resume()`` (which bypasses this
        method), so appending an episode had no route through the public API.
        """
        _seed_dataset(relocated_home / "user" / "ds")
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert _StubRecorder.calls == ["resume"], f"expected an append, got {_StubRecorder.calls}"

    def test_a_fresh_repo_id_is_still_created(self, relocated_home):
        """The mirror: nothing on disk stays a ``create``, not a resume."""
        engine = _engine()

        result = engine.start_recording(repo_id="user/ds", root=None, fps=30)

        assert result["status"] == "success", result
        assert _StubRecorder.calls == ["create"], f"expected a fresh dataset, got {_StubRecorder.calls}"

    def test_create_refuses_the_dataset_the_stale_probe_used_to_miss(self, relocated_home):
        """Pins the dead end itself, so the resume assertion above has a stated cost.

        This is the error a caller reached before the fix: the probe reported
        "nothing on disk", ``create()`` disagreed, and the message pointed away
        from ``start_recording``.
        """
        addressed = resolve_dataset_dir("user/ds", None)
        _seed_dataset(addressed)

        with pytest.raises(FileExistsError, match="already exists"):
            dr._prepare_create_target(addressed, overwrite=False)


_START_RECORDING_BACKENDS = (
    "strands_robots/simulation/mujoco/recording.py",
    "strands_robots/simulation/isaac/recording.py",
    "strands_robots/simulation/newton/recording.py",
)


def _called_names(module: str, method: str) -> set[str]:
    """Names of every plain ``f(...)`` call made inside ``module::method``."""
    tree = ast.parse(Path(module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    pytest.fail(f"{method} not found in {module}")


def _string_constants(module: str, method: str) -> set[str]:
    """Every string literal appearing inside ``module::method``."""
    tree = ast.parse(Path(module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    pytest.fail(f"{method} not found in {module}")


class TestNoDatasetRootResolutionDrifts:
    """Pinned structurally: the Isaac and Newton simulators cannot be driven here.

    Behavioural coverage above can only reach the backend whose recording
    lifecycle runs solver-free. These two assertions hold the property for the
    others, and for a backend added later -- which is how the hand-rolled copy
    survived: nothing said the resolution had one owner.
    """

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_every_backend_start_recording_uses_the_shared_resolver(self, module):
        assert "resolve_dataset_dir" in _called_names(module, "start_recording"), (
            f"{module}::start_recording resolves the dataset dir without resolve_dataset_dir"
        )

    @pytest.mark.parametrize("module", _START_RECORDING_BACKENDS)
    def test_no_backend_spells_the_dataset_home_itself(self, module):
        """A literal home component is the shape the drift took.

        ``resolve_dataset_dir`` reads the home from lerobot's own constant so the
        ``HF_LEROBOT_HOME`` override is honoured; a backend that spells any part
        of the default path has pinned it instead, which is what silently
        ignored the override.
        """
        literals = _string_constants(module, "start_recording")
        assert not literals & {".cache", "huggingface", "lerobot"}, (
            f"{module}::start_recording hard-codes the dataset home instead of resolving it"
        )
