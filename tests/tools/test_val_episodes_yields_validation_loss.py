"""``val_episodes`` must produce a validation signal, not just a smaller train set.

lerobot expresses a held-out validation split as ``dataset.eval_split`` (a
per-task fraction) and evaluates it every ``eval_steps`` training steps. The two
are coupled in lerobot's own ``TrainPipelineConfig.validate``: an ``eval_steps``
with no ``eval_split`` to draw held-out data from is a hard error, and an
``eval_split`` with ``eval_steps == 0`` builds the split but never evaluates it.

So a surface that reserves episodes by restricting ``dataset.episodes`` shrinks
the TRAINING set and produces no validation loss at all, and one that emits
``eval_steps`` alone refuses to launch. These tests pin that every surface taking
``val_episodes`` emits the pair, that the fraction reproduces the requested
episode COUNT exactly, and that a count which cannot be honored is refused
rather than silently rounded.
"""

import dataclasses
import importlib
import inspect
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.lerobot_train import build_train_command
from strands_robots.utils import (
    effective_episode_count,
    episode_subset_budget_error,
    validation_split_error,
    validation_split_fraction,
)


def _importable(module: str, symbol: str | None = None) -> bool:
    """Whether ``from module import symbol`` (or ``import module``) would succeed here."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    return symbol is None or hasattr(mod, symbol)


# lerobot 0.6.1 (the declared floor) has neither resolve_episode_indices nor
# DatasetConfig.exclude_episodes; both landed in a single commit (64b23178d).
# Cells that assert resolver semantics (exclusion lists, allowlist+exclusion,
# out-of-range index shrinkage) are gated on its presence so the required check
# passes on the locked environment.
_HAS_RESOLVER = _importable("lerobot.datasets.utils", "resolve_episode_indices")

_NEEDS_RESOLVER = pytest.mark.skipif(
    not _HAS_RESOLVER,
    reason="resolve_episode_indices absent in locked lerobot 0.6.1",
)

_HAS_DRACCUS = _importable("draccus.cfgparsing")

_NEEDS_DRACCUS = pytest.mark.skipif(
    not _HAS_DRACCUS,
    reason="draccus (lerobot's CLI decoder for text-form episode lists) not installed",
)


def _write_dataset(root: Path, total_episodes: int = 10, total_tasks: int = 1) -> Path:
    """Minimal LeRobot v3 dataset stub carrying episode and task counts."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": total_episodes, "total_tasks": total_tasks}))
    return root


def _flag(cmd: list[str], name: str) -> str | None:
    """The single value emitted for ``--name=``, or None when absent."""
    hits = [c.split("=", 1)[1] for c in cmd if c.startswith(f"--{name}=")]
    assert len(hits) <= 1, f"--{name} emitted {len(hits)} times: {hits}"
    return hits[0] if hits else None


class TestFractionReproducesRequestedCount:
    """The emitted fraction must hold out exactly the requested episode count."""

    # (total, N) pairs where the obvious ``N / total`` fraction is NOT float-safe:
    # 25 * (7 / 25) == 7.000000000000001, whose ceiling is 8 - one episode more
    # than asked. These are the cases a boundary fraction gets wrong.
    @pytest.mark.parametrize(("total", "n"), [(25, 7), (25, 14), (29, 15), (35, 29), (38, 21), (41, 7)])
    def test_boundary_unsafe_pairs_still_reserve_exactly_n(self, total: int, n: int) -> None:
        assert math.ceil(total * (n / total)) != n, "fixture no longer exercises the float hazard"
        assert math.ceil(total * validation_split_fraction(n, total)) == n

    @pytest.mark.parametrize("total", [2, 3, 10, 50, 206, 397])
    def test_every_count_reserves_exactly_n(self, total: int) -> None:
        """Sweep every reservable count, not just the hand-picked hazards."""
        for n in range(1, total):
            assert math.ceil(total * validation_split_fraction(n, total)) == n, (total, n)

    def test_command_fraction_reserves_exactly_n(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=5)
        assert math.ceil(50 * float(_flag(cmd, "dataset.eval_split") or "0")) == 5


class TestEmittedFlagsCanProduceAValidationLoss:
    """Both halves of lerobot's coupled pair must be emitted together."""

    def test_val_episodes_emits_split_and_a_nonzero_eval_cadence(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", save_freq=250, val_episodes=2)
        assert float(_flag(cmd, "dataset.eval_split") or "0") > 0.0
        assert int(_flag(cmd, "eval_steps") or "0") == 250

    def test_eval_cadence_is_never_emitted_without_a_split_to_evaluate(self, tmp_path: Path) -> None:
        """lerobot rejects eval_steps > 0 when eval_split is 0.0, so never pair them that way."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        # Annotated so mypy does not narrow the splat to the inferred value type.
        cases: list[dict[str, Any]] = [{}, {"val_episodes": 3}]
        for kwargs in cases:
            cmd = build_train_command(dataset_root=str(root), policy_type="act", **kwargs)
            steps = int(_flag(cmd, "eval_steps") or "0")
            split = float(_flag(cmd, "dataset.eval_split") or "0")
            assert (steps > 0) == (split > 0.0), f"{kwargs}: eval_steps={steps} eval_split={split}"

    def test_no_val_episodes_leaves_evaluation_untouched(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act")
        assert _flag(cmd, "dataset.eval_split") is None
        assert _flag(cmd, "eval_steps") is None

    def test_disabled_periodic_saving_still_evaluates_once(self, tmp_path: Path) -> None:
        """save_freq <= 0 disables checkpointing, so fall back to a pass at the end."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", steps=4000, save_freq=0, val_episodes=2)
        assert int(_flag(cmd, "eval_steps") or "0") == 4000

    def test_reserved_episodes_are_not_also_cut_from_training_by_hand(self, tmp_path: Path) -> None:
        """A hand-rolled --dataset.episodes restriction would hide the split from lerobot."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2)
        assert _flag(cmd, "dataset.episodes") is None


class TestCountThatCannotBeHonoredIsRefused:
    """A per-task fraction cannot express a global count on a multi-task dataset."""

    def test_multi_task_dataset_is_refused_with_the_fraction_to_use(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10, total_tasks=3)
        with pytest.raises(ValueError, match="cannot be reserved exactly"):
            build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2)

    @pytest.mark.parametrize("tasks", [0, 1, None])
    def test_absent_or_single_task_count_is_honored(self, tasks: object) -> None:
        """0 / None mean 'no task count recorded', which lerobot treats as one task.

        ``True`` and ``"2"`` are NOT in this list: a header declaring something
        which is not a task count is a third outcome, refused on its own terms
        rather than read as an absent one, and pinned in
        ``tests/test_declared_task_count_has_one_owner.py``.
        """
        assert validation_split_error(2, tasks, "ctx", passthrough_param="extra_flags") is None

    def test_refusal_names_the_task_count_and_the_direct_knobs(self) -> None:
        err = validation_split_error(1, 4, "lerobot_train", passthrough_param="extra_flags")
        assert err is not None
        assert err.startswith("lerobot_train: ")
        assert "4 tasks" in err
        assert "dataset.eval_split" in err and "eval_steps" in err


class TestCallerOverridesTakePrecedence:
    """extra_flags is the documented escape hatch; it must not be duplicated."""

    @pytest.mark.parametrize("key", ["eval_steps", "dataset.eval_split"])
    def test_explicit_value_replaces_the_derived_one(self, tmp_path: Path, key: str) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2, extra_flags={key: 7})
        # _flag asserts the flag appears at most once, so a duplicate fails here.
        assert _flag(cmd, key) == "7"


def _held_out(cmd: list[str], loaded_episodes: int) -> int:
    """Episodes lerobot will hold out of the emitted split, for a single task.

    Mirrors ``lerobot.datasets.factory.make_train_eval_datasets``, which takes
    ``ceil(len(eps) * eval_split)`` from the tail of the episodes the DATASET was
    built from - the subset ``dataset.episodes`` / ``dataset.exclude_episodes``
    left, not the header's ``total_episodes``.
    """
    split = _flag(cmd, "dataset.eval_split")
    return 0 if split is None else math.ceil(loaded_episodes * float(split))


# Every spelling of "load 15 of the 30 episodes" a passthrough accepts, with the
# episode count each one leaves the run. The text form is the one lerobot's own
# CLI decoder reads, so it reaches training exactly as the list does.
_FIFTEEN_OF_THIRTY: list[Any] = [
    ("allowlist", {"dataset.episodes": list(range(15))}, 15),
    pytest.param("allowlist as text", {"dataset.episodes": str(list(range(15)))}, 15, marks=_NEEDS_DRACCUS),
    # --- resolver-dependent: exclusion lists and out-of-range shrinkage ---
    pytest.param("exclusion list", {"dataset.exclude_episodes": list(range(15, 30))}, 15, marks=_NEEDS_RESOLVER),
    pytest.param(
        "allowlist and exclusion",
        {"dataset.episodes": list(range(20)), "dataset.exclude_episodes": [0, 1, 2, 3, 4]},
        15,
        marks=_NEEDS_RESOLVER,
    ),
    # lerobot's resolver drops an index outside the dataset with a warning
    # instead of refusing it, so an out-of-range entry SHRINKS the subset.
    pytest.param(
        "allowlist with an out-of-range index", {"dataset.episodes": [*range(15), 99]}, 15, marks=_NEEDS_RESOLVER
    ),
]


class TestTheSplitIsSizedAgainstTheLoadedSubset:
    """The requested COUNT must survive an episode subset, on every surface.

    ``val_episodes`` is delivered as one ``eval_split`` FRACTION, and lerobot
    multiplies that fraction by the episodes the dataset was BUILT from. Dividing
    it by the header's ``total_episodes`` while lerobot multiplies by the subset
    reserved fewer episodes than asked - 2 of the 15 episodes an episode filter
    kept, for a caller who asked for 3 - and the run still logged an eval loss,
    so it looked correct. This is the documented ``filter_episodes`` recipe in
    ``docs/data/episode-labels.md``.
    """

    @pytest.mark.parametrize(
        ("label", "passthrough", "loaded"),
        _FIFTEEN_OF_THIRTY,
    )
    def test_the_tool_reserves_exactly_n_of_the_subset(
        self, tmp_path: Path, label: str, passthrough: dict[str, Any], loaded: int
    ) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        cmd = build_train_command(
            dataset_root=str(root), policy_type="act", val_episodes=3, extra_flags=dict(passthrough)
        )
        assert _held_out(cmd, loaded) == 3

    @pytest.mark.parametrize(
        ("label", "passthrough", "loaded"),
        _FIFTEEN_OF_THIRTY,
    )
    def test_the_trainspec_backend_reserves_exactly_n_of_the_subset(
        self, tmp_path: Path, label: str, passthrough: dict[str, Any], loaded: int
    ) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        spec = _spec(dataset_root=str(root), val_episodes=3, extra={"policy_type": "act", **passthrough})
        trainer = LerobotTrainer(device="cpu")
        assert _held_out(trainer.build_command(spec), loaded) == 3
        # The in-process config must carry the same fraction the argv does, or
        # the two launch paths reserve different numbers of episodes.
        assert trainer.build_config(spec).dataset.eval_split == pytest.approx(validation_split_fraction(3, loaded))

    def test_a_hydra_prefixed_key_names_the_same_subset(self, tmp_path: Path) -> None:
        """``--dataset.episodes`` and ``dataset.episodes`` are one flag in the argv."""
        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        cmd = build_train_command(
            dataset_root=str(root),
            policy_type="act",
            val_episodes=3,
            extra_flags={"--dataset.episodes": list(range(15))},
        )
        assert _held_out(cmd, 15) == 3

    def test_no_subset_still_divides_by_the_whole_dataset(self, tmp_path: Path) -> None:
        """The fix must not shrink a denominator nothing restricted."""
        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=3)
        assert _held_out(cmd, 30) == 3


class TestASubsetTooSmallForTheHoldoutIsRefused:
    """The holdout is bounded by the episodes loaded, not by the header count.

    ``val_episodes=5`` against a 30-episode header whose passthrough selected 4
    cleared the whole-dataset bound check and emitted a fraction that held out 1.
    """

    def test_the_tool_refuses_and_names_both_counts(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        with pytest.raises(ValueError, match="cannot be reserved from the 4 episode") as excinfo:
            build_train_command(
                dataset_root=str(root),
                policy_type="act",
                val_episodes=5,
                extra_flags={"dataset.episodes": [0, 1, 2, 3]},
            )
        assert "out of 30 in the dataset" in str(excinfo.value)

    def test_the_trainspec_backend_refuses_and_names_both_counts(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        spec = _spec(
            dataset_root=str(root),
            val_episodes=5,
            extra={"policy_type": "act", "dataset.episodes": [0, 1, 2, 3]},
        )
        problems = LerobotTrainer(device="cpu").validate(spec)
        named = [p for p in problems if "cannot be reserved from the 4 episode" in p]
        assert named, problems
        assert "out of 30 in the dataset" in named[0]

    @_NEEDS_RESOLVER
    def test_an_out_of_range_allowlist_is_refused_here_not_at_train_time(self, tmp_path: Path) -> None:
        """lerobot drops an out-of-range index, so the subset can be too small.

        The resolver warns rather than raising, so the shrunken subset is what
        reaches the split - and a holdout that no longer fits it is refused by
        the same bound check, before a run starts.
        """
        root = _write_dataset(tmp_path / "ds", total_episodes=30)
        with pytest.raises(ValueError, match="cannot be reserved from the 3 episode"):
            build_train_command(
                dataset_root=str(root),
                policy_type="act",
                val_episodes=3,
                extra_flags={"dataset.episodes": [0, 1, 2, 99]},
            )

    def test_the_whole_dataset_refusal_is_unchanged_without_a_subset(self, tmp_path: Path) -> None:
        """No subset means the caller's own header-count refusal still answers."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        with pytest.raises(ValueError, match="leaves no training data"):
            build_train_command(dataset_root=str(root), policy_type="act", val_episodes=10)
        assert episode_subset_budget_error(10, 10, 10, "ctx", passthrough_param="extra") is None


class TestTheLoadedEpisodeCountHasOneOwner:
    """Both surfaces read the subset through :func:`effective_episode_count`."""

    @pytest.mark.parametrize(
        ("episodes", "exclude", "expected"),
        [
            (None, None, 30),
            (None, [], 30),
            (list(range(15)), None, 15),
            # --- resolver-dependent: the floor ignores exclusion / counts verbatim ---
            pytest.param(None, list(range(15, 30)), 15, marks=_NEEDS_RESOLVER),
            pytest.param(list(range(20)), [0, 1, 2, 3, 4], 15, marks=_NEEDS_RESOLVER),
            pytest.param([0, 1, 2, 99], None, 3, marks=_NEEDS_RESOLVER),
            # --- floor-path expectations (no resolver: exclusion ignored, allowlist verbatim) ---
            pytest.param(None, list(range(15, 30)), 30, marks=pytest.mark.skipif(_HAS_RESOLVER, reason="floor-only")),
            pytest.param(
                list(range(20)), [0, 1, 2, 3, 4], 20, marks=pytest.mark.skipif(_HAS_RESOLVER, reason="floor-only")
            ),
            pytest.param([0, 1, 2, 99], None, 4, marks=pytest.mark.skipif(_HAS_RESOLVER, reason="floor-only")),
            pytest.param("[0, 1, 2]", None, 3, marks=_NEEDS_DRACCUS),
            ([], None, 0),
            # Values the config field itself refuses are no restriction here:
            # no run starts on them, so there is no split to size. A bool is an
            # int in Python, and reading True as "episode 1" would size a split
            # against one episode.
            (True, None, 30),
            ("not-a-list", None, 30),
            ([0, 1.5], None, 30),
            ([0, True], None, 30),
        ],
    )
    def test_the_loaded_count_is_what_lerobot_will_build_the_dataset_from(
        self, episodes: Any, exclude: Any, expected: int
    ) -> None:
        assert effective_episode_count(30, episodes, exclude) == expected

    @pytest.mark.parametrize(
        ("hostile", "expected"),
        [
            pytest.param("iter", 30, id="__iter__ raises - no restriction"),
            pytest.param("next", 30, id="an element raises mid-read - no restriction"),
            # The read iterates rather than measuring, so a value with no usable
            # length is still read: 3 indices is the honest count, not a fallback.
            pytest.param("len", 3, id="__len__ raises - still read by iteration"),
        ],
    )
    def test_a_sequence_that_will_not_be_read_does_not_escape_the_preflight(self, hostile: str, expected: int) -> None:
        """The count feeds ``validate()``, which must return a verdict, not raise.

        A ``Sequence`` is only a promise; reading one runs the caller's code. So
        the read is guarded element by element and a value that cannot be read
        counts as no restriction - the config field refuses it with the spellings
        it accepts.
        """

        class Hostile(Sequence):  # type: ignore[type-arg]
            def __iter__(self) -> Any:
                if hostile == "iter":
                    raise RuntimeError("no iteration for you")
                if hostile == "next":

                    def produce() -> Any:
                        yield 0
                        raise RuntimeError("no second element for you")

                    return produce()
                return iter([0, 1, 2])

            def __getitem__(self, index: Any) -> Any:
                raise RuntimeError("no subscript for you")

            def __len__(self) -> int:
                if hostile == "len":
                    raise RuntimeError("no length for you")
                return 3

        assert effective_episode_count(30, Hostile(), None) == expected

    def test_a_lerobot_without_the_resolver_counts_the_allowlist_verbatim(self, monkeypatch: Any) -> None:
        """lerobot 0.6.1, the declared floor, has no ``resolve_episode_indices``.

        It also has no ``DatasetConfig.exclude_episodes`` (both landed in one
        commit) and hands ``dataset.episodes`` to ``LeRobotDataset`` unchanged,
        so the allowlist length is the loaded count there.
        """
        import builtins

        real_import = builtins.__import__

        def refuse_resolver(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lerobot.datasets.utils":
                raise ImportError("no resolve_episode_indices on the floor")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_resolver)
        assert effective_episode_count(30, list(range(15)), None) == 15
        assert effective_episode_count(30, None, None) == 30

    def test_a_multi_task_dataset_is_still_refused_with_a_subset(self, tmp_path: Path) -> None:
        """A subset can only narrow the task set, so the header guard stays a superset.

        ``validation_split_error`` reads ``total_tasks`` from the header rather
        than over the subset. Since the subset's tasks are a subset of the
        dataset's, a single-task header still means a single-task subset, and a
        multi-task header is refused either way - never silently honored.
        """
        root = _write_dataset(tmp_path / "ds", total_episodes=30, total_tasks=3)
        with pytest.raises(ValueError, match="cannot be reserved exactly"):
            build_train_command(
                dataset_root=str(root),
                policy_type="act",
                val_episodes=3,
                extra_flags={"dataset.episodes": list(range(15))},
            )


class TestEverySurfaceAgrees:
    """The tool and the TrainSpec backend must not drift on one knob."""

    def test_trainspec_path_emits_the_same_pair(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        spec = TrainSpec(dataset_root=str(root), output_dir=str(tmp_path / "out"), val_episodes=5, save_freq=250)
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert math.ceil(50 * float(_flag(cmd, "dataset.eval_split") or "0")) == 5
        assert int(_flag(cmd, "eval_steps") or "0") == 250
        assert _flag(cmd, "dataset.episodes") is None

    def test_trainspec_validate_refuses_a_multi_task_count(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=10, total_tasks=3)
        spec = TrainSpec(dataset_root=str(root), output_dir=str(tmp_path / "out"), val_episodes=2)
        assert any("cannot be reserved exactly" in p for p in LerobotTrainer().validate(spec))


def _spec(**kwargs: Any) -> Any:
    """A launchable :class:`TrainSpec`, overridden per case."""
    from strands_robots.training.base import TrainSpec

    return TrainSpec(output_dir="/tmp/strands-val-episodes-out", steps=10, save_freq=5, **kwargs)


def _passthrough_keyword(message: str) -> str:
    """The passthrough parameter a refusal tells the reader to use instead."""
    match = re.search(r"(\w+)=\{'dataset\.eval_split'", message)
    assert match is not None, f"refusal names no dataset.eval_split passthrough: {message}"
    return match.group(1)


class TestAnUnreadableEpisodeCountIsRefused:
    """A count that cannot become a fraction must be refused, not dropped.

    The split is ``ceil(episodes_in_task * eval_split)``, so turning an episode
    COUNT into lerobot's fraction needs the dataset's ``total_episodes``. That
    number is only in a local ``meta/info.json``, which a Hub source need not
    have: ``dataset_repo_id`` with no ``dataset_root``, or a ``dataset_root``
    that is a Hub cache directory nothing has been downloaded into yet. Emitting
    no fraction there is indistinguishable from "no validation was asked for" -
    the run trains on every episode and records no validation loss, which is the
    outcome this module exists to prevent.
    """

    def test_a_hub_source_with_no_local_root_is_refused(self) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        spec = _spec(dataset_repo_id="lerobot/svla_so101_pickplace", val_episodes=2)
        trainer = LerobotTrainer()
        problems = trainer.validate(spec)
        assert any("val_episodes=2" in p for p in problems), (
            "a Hub source drops val_episodes silently: validate() reported "
            f"{problems} while build_command() emits "
            f"{[c for c in trainer.build_command(spec) if 'eval' in c]}"
        )

    def test_a_hub_cache_root_with_nothing_downloaded_is_refused(self, tmp_path: Path) -> None:
        """The documented Hub-cache shape: repo id plus an empty local cache dir."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        cache = tmp_path / "hf-cache"
        cache.mkdir()
        spec = _spec(dataset_repo_id="lerobot/svla_so101_pickplace", dataset_root=str(cache), val_episodes=2)
        assert any("val_episodes=2" in p for p in LerobotTrainer().validate(spec))

    def test_the_refusal_is_load_bearing_so_no_run_launches_without_the_split(self) -> None:
        """``train`` fails closed on validate(), so the request cannot be dropped."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        result = LerobotTrainer().train(_spec(dataset_repo_id="lerobot/svla_so101_pickplace", val_episodes=2))
        assert result.status != "success"
        assert "val_episodes=2" in result.message


class TestARefusalNamesAPassthroughItsOwnSurfaceAccepts:
    """Every remedy pointing at raw flags must name a keyword that surface has.

    The passthrough is spelled ``extra_flags`` on the ``lerobot_train`` tool and
    ``extra`` on :class:`TrainSpec` (and the ``train_policy`` tool), so one
    hardcoded spelling makes the remedy a dead end on the other surface: applied
    verbatim it raises ``TypeError`` for an unexpected keyword.
    """

    def test_the_trainer_names_a_real_trainspec_field(self) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        problems = LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x", val_episodes=2))
        keyword = _passthrough_keyword(" ".join(problems))
        assert keyword in {f.name for f in dataclasses.fields(TrainSpec)}

    def test_the_tool_names_a_real_lerobot_train_parameter(self) -> None:
        from strands_robots.tools.lerobot_train import lerobot_train

        err = validation_split_error(1, 4, "lerobot_train", passthrough_param="extra_flags")
        assert err is not None
        target = getattr(lerobot_train, "__wrapped__", lerobot_train)
        assert _passthrough_keyword(err) in inspect.signature(target).parameters

    def test_applying_the_trainer_remedy_verbatim_produces_the_pair(self) -> None:
        """Parse the remedy out of the refusal, apply it, and it must work."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        trainer = LerobotTrainer()
        keyword = _passthrough_keyword(" ".join(trainer.validate(_spec(dataset_repo_id="lerobot/x", val_episodes=2))))
        remedied = _spec(dataset_repo_id="lerobot/x", **{keyword: {"dataset.eval_split": 0.1, "eval_steps": 5}})
        assert trainer.validate(remedied) == []
        cmd = trainer.build_command(remedied)
        assert _flag(cmd, "dataset.eval_split") == "0.1"
        assert _flag(cmd, "eval_steps") == "5"

    def test_pointing_dataset_root_at_a_local_copy_is_the_other_named_remedy(self, tmp_path: Path) -> None:
        """The refusal's first remedy: a populated cache root makes the count readable."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "cache", total_episodes=10)
        spec = _spec(dataset_repo_id="lerobot/x", dataset_root=str(root), val_episodes=2)
        trainer = LerobotTrainer()
        assert trainer.validate(spec) == []
        assert math.ceil(10 * float(_flag(trainer.build_command(spec), "dataset.eval_split") or "0")) == 2


class TestTheRefusalDoesNotReachPastTheDefect:
    """Controls: only an unhonorable request may be refused."""

    def test_a_local_root_still_emits_the_pair(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        spec = _spec(dataset_root=str(root), val_episodes=5)
        trainer = LerobotTrainer()
        assert trainer.validate(spec) == []
        assert math.ceil(50 * float(_flag(trainer.build_command(spec), "dataset.eval_split") or "0")) == 5

    def test_a_hub_source_without_val_episodes_is_not_refused(self) -> None:
        """None is the documented "train on everything" sentinel, not a problem."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        assert LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x")) == []

    def test_an_unusable_count_still_names_the_count_domain_first(self) -> None:
        """A non-positive count is a domain problem wherever the data lives."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        problems = LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x", val_episodes=0))
        assert any("must be a positive integer" in p for p in problems)
