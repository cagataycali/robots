"""A fresh training start clears an EMPTY ``output_dir``, and nothing else.

Both LeRobot training entry points clear a stale ``output_dir`` on a fresh
(non-resuming) start, because lerobot's own ``TrainPipelineConfig.validate``
refuses a pre-existing one unless ``resume=True``::

    FileExistsError: Output directory <dir> already exists and resume is False.
    Please change your output directory so that <dir> is not overwritten.

The two implemented the same rule with different bounds. ``lerobot_train``
required the directory to be EMPTY and said why ("Never delete a dir that holds
checkpoints"); :meth:`LerobotTrainer.train` required only that no *resumable
checkpoint* was visible, and cleared whatever else was there with a recursive
``shutil.rmtree(..., ignore_errors=True)`` that reports neither what it took nor
a partial failure.

"No resumable checkpoint" is a strictly weaker bound than "empty", and the gap
is where a caller's data lives:

* A checkpoint a resume probe cannot see. lerobot's ``save_checkpoint`` writes
  ``model.safetensors`` before ``train_config.json``, so a run interrupted
  between the two leaves the trained weights on disk under a checkpoint that
  answers "not resumable" -- and a checkpoint-keyed bound clears exactly that.
* A directory that is not a run at all. ``output_dir`` is a caller-supplied
  path on both entry points, including the ``train_policy`` agent tool.

The removal also ran ahead of refusals that follow it: ``build_config`` raises
on an unusable ``extra`` after the hygiene step, so the call could take the
directory and then report an error, having trained nothing.

Emptiness SUBSUMES the checkpoint probe -- a directory holding a checkpoint is
not empty, whatever layout the checkpoint is in -- so the shared owner needs no
checkpoint probe, and the ``lerobot_train`` side keeps its previous verdict on
every input while dropping the redundant test.
"""

import ast
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as tool_mod
import strands_robots.training.lerobot as trainer_mod
from strands_robots.tools.lerobot_train import lerobot_train
from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.utils import stale_output_dir_is_clearable

# The directory shapes an ``output_dir`` can be in when a fresh start looks at
# it. ``clearable`` is the verdict BOTH entry points must reach.
#   empty              - the only shape a recursive removal costs nothing on
#   leftover-file      - a stale run's logs/plots, or a directory that is
#                        not a run at all
#   intact-checkpoint  - a checkpoint a resume probe reports
#   headless-checkpoint- lerobot writes model.safetensors BEFORE
#                        train_config.json, so an interrupted save leaves the
#                        trained weights under a checkpoint no resume probe
#                        reports
_SHAPES: tuple[tuple[str, bool], ...] = (
    ("empty", True),
    ("leftover-file", False),
    ("leftover-subdir", False),
    ("intact-checkpoint", False),
    ("headless-checkpoint", False),
)


def _make(root: Path, shape: str) -> Path:
    """Materialise one ``_SHAPES`` entry at ``root`` and return it."""
    root.mkdir(parents=True, exist_ok=True)
    if shape == "empty":
        return root
    if shape == "leftover-file":
        (root / "loss.png").write_bytes(b"\x89PNG stale plot")
        return root
    if shape == "leftover-subdir":
        run = root / "wandb" / "run-1"
        run.mkdir(parents=True)
        # A FILE, not just the directories: the census below counts files, so a
        # tree of empty directories would make this row grade nothing.
        (run / "output.log").write_text("step 1 loss 0.4\n")
        return root
    ckpt = root / "checkpoints" / "000100" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"trained weights")
    (ckpt / "config.json").write_text("{}")
    if shape == "intact-checkpoint":
        (ckpt / "train_config.json").write_text("{}")
        (root / "checkpoints" / "last").symlink_to(ckpt.parent, target_is_directory=True)
    elif shape != "headless-checkpoint":  # pragma: no cover - guards the table
        raise AssertionError(f"unknown shape {shape!r}")
    return root


def _survivors(root: Path) -> list[str]:
    """Every file still under ``root``, relative and sorted."""
    if not root.exists():
        return []
    return sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file())


def _write_dataset(root: Path) -> Path:
    """A minimal LeRobot v3 dataset root: what both entry points read."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 4}))
    return root


class _FakeProc:
    """Stands in for the detached training process the tool launches."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tool_mod, "SESSION_DIR", session_dir)


class TestTheSharedBoundIsEmptiness:
    """The owner answers True for an empty directory and nothing else."""

    @pytest.mark.parametrize(("shape", "clearable"), _SHAPES, ids=[s for s, _ in _SHAPES])
    def test_only_an_empty_directory_is_clearable(self, tmp_path: Path, shape: str, clearable: bool) -> None:
        out = _make(tmp_path / "out", shape)
        assert stale_output_dir_is_clearable(str(out)) is clearable

    def test_a_path_that_does_not_exist_is_not_clearable(self, tmp_path: Path) -> None:
        """Nothing to clear, so the caller must not be told there is."""
        assert stale_output_dir_is_clearable(str(tmp_path / "absent")) is False

    def test_a_file_is_not_clearable(self, tmp_path: Path) -> None:
        """A non-directory is never a run's output_dir; leave it to lerobot."""
        target = tmp_path / "not-a-dir"
        target.write_text("data")
        assert stale_output_dir_is_clearable(str(target)) is False

    def test_emptiness_subsumes_the_checkpoint_probe(self, tmp_path: Path) -> None:
        """Premise for dropping the probe: an empty dir holds no checkpoint.

        Holds on both trees by construction - it is why the shared owner needs
        no checkpoint probe of its own, not a claim about the fix.
        """
        empty = _make(tmp_path / "empty", "empty")
        assert tool_mod._has_resumable_checkpoint(str(empty)) is None
        assert LerobotTrainer(device="cpu").latest_checkpoint(str(empty)) is None

    def test_an_interrupted_save_hides_the_weights_from_both_resume_probes(self, tmp_path: Path) -> None:
        """Premise: the weights are there and neither probe reports them.

        Also holds on both trees - it establishes that a checkpoint-keyed bound
        and an emptiness bound genuinely disagree on this shape.
        """
        out = _make(tmp_path / "out", "headless-checkpoint")
        assert (out / "checkpoints" / "000100" / "pretrained_model" / "model.safetensors").exists()
        assert tool_mod._has_resumable_checkpoint(str(out)) is None
        assert LerobotTrainer(device="cpu").latest_checkpoint(str(out)) is None


class TestBothEntryPointsReachTheSameVerdict:
    """One rule, two entry points, one verdict per directory shape."""

    @staticmethod
    def _trainer_verdict(tmp_path: Path, shape: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        out = _make(tmp_path / "trainer_out", shape)
        spec = TrainSpec(
            dataset_root=str(_write_dataset(tmp_path / "ds")),
            output_dir=str(out),
            steps=2,
            global_batch_size=2,
            save_freq=1,
            extra={"policy_type": "act"},
        )
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())
        import lerobot.scripts.lerobot_train as lt

        monkeypatch.setattr(lt, "train", lambda cfg, **kw: None)
        trainer.train(spec)
        return _survivors(out)

    @staticmethod
    def _tool_verdict(tmp_path: Path, shape: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        out = _make(tmp_path / "tool_out", shape)
        monkeypatch.setattr(tool_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
        result = lerobot_train(
            action="start",
            dataset_root=str(_write_dataset(tmp_path / "ds")),
            output_dir=str(out),
            policy_type="act",
            session_name=f"parity_{shape}",
        )
        assert result["status"] == "success", result
        return _survivors(out)

    @pytest.mark.parametrize(("shape", "clearable"), _SHAPES, ids=[s for s, _ in _SHAPES])
    def test_the_two_entry_points_keep_the_same_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, clearable: bool
    ) -> None:
        expected = [] if clearable else _survivors(_make(tmp_path / "reference", shape))
        assert self._trainer_verdict(tmp_path, shape, monkeypatch) == expected
        assert self._tool_verdict(tmp_path, shape, monkeypatch) == expected

    def test_the_trainer_still_clears_an_empty_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hygiene the bound exists to allow still happens.

        Control: a bound that refused every shape would satisfy the rows above
        and break the reason the removal is there at all.
        """
        out = _make(tmp_path / "trainer_out", "empty")
        self._trainer_verdict(tmp_path, "empty", monkeypatch)
        assert not out.exists()

    def test_a_resuming_start_clears_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``resume=True`` keeps the checkpoints the run is resuming from."""
        out = _make(tmp_path / "trainer_out", "intact-checkpoint")
        before = _survivors(out)
        spec = TrainSpec(
            dataset_root=str(_write_dataset(tmp_path / "ds")),
            output_dir=str(out),
            steps=2,
            global_batch_size=2,
            save_freq=1,
            resume=True,
            extra={"policy_type": "act"},
        )
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())
        import lerobot.scripts.lerobot_train as lt

        monkeypatch.setattr(lt, "train", lambda cfg, **kw: None)
        trainer.train(spec)
        assert _survivors(out) == before


class TestARefusalAfterTheHygieneCostsNothing:
    """A refusal that follows the removal must not have cost the caller data."""

    def test_a_build_config_failure_leaves_a_non_empty_directory_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _make(tmp_path / "out", "headless-checkpoint")
        before = _survivors(out)
        spec = TrainSpec(
            dataset_root=str(_write_dataset(tmp_path / "ds")),
            output_dir=str(out),
            steps=2,
            global_batch_size=2,
            save_freq=1,
            extra={"policy_type": "act"},
        )
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])

        def _refuse(_spec: TrainSpec) -> Any:
            raise ValueError("extra['sample_weighting'] does not support field(s) ['no_such_field']")

        monkeypatch.setattr(trainer, "build_config", _refuse)
        result = trainer.train(spec)

        assert result.status == "error"
        assert "failed to build lerobot TrainPipelineConfig" in result.message
        assert _survivors(out) == before, "a refused call removed the caller's files"


class TestTheBoundHasOneOwner:
    """Neither entry point re-implements the bound beside its removal."""

    @pytest.mark.parametrize("module", (trainer_mod, tool_mod), ids=("trainer", "tool"))
    def test_every_output_dir_removal_is_guarded_by_the_shared_owner(self, module: Any) -> None:
        """Read the NEAREST enclosing ``if`` of each removal, not any of them.

        Graded on the call rather than the source text so a second copy of the
        bound cannot drift back in beside a removal under another name.
        """
        tree = ast.parse(inspect.getsource(module))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        removals = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "rmtree"
        ]
        assert len(removals) == 1, f"{module.__name__}: expected one shutil.rmtree, found {len(removals)}"

        for removal in removals:
            guard: ast.AST | None = removal
            while guard is not None and not isinstance(guard, ast.If):
                guard = parents.get(guard)
            assert isinstance(guard, ast.If), f"{module.__name__}: the removal is not guarded at all"
            names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
            assert "stale_output_dir_is_clearable" in names, (
                f"{module.__name__}: a shutil.rmtree of the output dir is guarded by "
                f"{ast.unparse(guard.test)!r} instead of the shared bound"
            )

    def test_the_removal_is_still_recursive_and_silent(self) -> None:
        """Premise for the bound: what makes it unsafe on a non-empty dir.

        Holds on both trees - the point is the bound, not the removal, and a
        reader needs to know the removal reports nothing.
        """
        assert shutil.rmtree is trainer_mod.shutil.rmtree
        source = inspect.getsource(trainer_mod.LerobotTrainer.train)
        assert "ignore_errors=True" in source
