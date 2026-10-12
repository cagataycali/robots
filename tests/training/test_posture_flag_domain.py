"""``resume`` and ``streaming`` on a TrainSpec are checked, never read by truthiness.

Each of the two selects a *posture* - continue the run under ``output_dir`` or
start it fresh; stream the dataset or materialize it - and every backend that
read either did so by truthiness. Every non-empty string is truthy, so the
spellings a caller reaches for when opting out (``"false"``, ``"no"``, ``"0"``)
selected the affirmative posture, and ``0``, ``None`` and ``""`` selected the
negative one without being a declared spelling of it. Measured on ``d2f23d8``:

* ``TrainSpec(resume="false")`` with a checkpoint under ``output_dir`` made
  LeRobot's ``build_config`` return the checkpoint's own ``train_config.json``
  in place of a config built from the spec, so the fresh run that was asked for
  became a resume of a previous configuration and the spec's ``steps``,
  ``global_batch_size`` and ``save_freq`` never reached it - the defect
  ``tests/tools/test_lerobot_train_flag_domain.py`` closed on the sibling tool,
  still reachable through the provider-agnostic ``train_policy`` one layer up;
* ``TrainSpec(streaming="false", val_episodes=2)`` was refused by LeRobot's
  ``validate`` with "set streaming=False to keep the validation split", a remedy
  the caller had already spelled - the flag *gates* the pair check, so a
  misread posture was reported as the option it selected rather than as itself;
* GR00T emitted ``--resume_from_checkpoint`` for ``resume="false"``, and
  SageMaker forwarded the raw string as a hyperparameter.

The fields are held to the shared :func:`~strands_robots.utils.boolean_flag_error`
domain through two field-scoped gates on :class:`~strands_robots.training.base.Trainer`,
in the same biconditional as the numeric domains beside them: a backend that
reads the field routes it through the gate, and a backend that ignores the field
reports nothing about it. Two gates rather than one because the readers differ -
GR00T reads ``resume`` and ignores ``streaming``.

Which backends refuse which values is the table in
``tests/training/test_train_spec_field_domains.py``; the tests below pin the
refusal's order against the check the flag gates and the provider-agnostic tool.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from strands_robots.tools.train_policy import train_policy
from strands_robots.training.base import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

# The spellings a caller reaches for when opting out (every one truthy), two
# truthy numbers, and the falsy values that are not a declared spelling of the
# negative posture either.

# Both python spellings plus the numpy booleans the shared domain also accepts.

# Which backend reads which field, by name or through a forwarding table. The
# one-owner scan at the bottom derives the same sets from the tree, so a
# backend that starts reading a field is graded on arrival.


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose posture flags are the only thing under test.

    ``validate`` may report unrelated problems (Cosmos wants a recipe TOML,
    SageMaker an S3 URI); every assertion below filters for the field name, so
    an unrelated problem can neither mask nor fake a verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


def _set(spec: TrainSpec, **fields: Any) -> TrainSpec:
    """Assign posture values through one funnel.

    The values under test are deliberately outside the declared ``bool``, which
    is the point; routing the assignment through ``setattr`` states that once
    rather than suppressing it at every site.
    """
    for name, value in fields.items():
        setattr(spec, name, value)
    return spec


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel, for the same reason as _set."""
    return dict(train_policy(**kwargs))


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


class TestTheRefusalPrecedesTheCheckTheFlagGates:
    """LeRobot's streaming / val_episodes pair check branches on the flag.

    A posture guard placed *after* that check still refuses, but names the
    option the misread posture selected rather than the flag - "set
    streaming=False" at a caller who had spelled exactly that. So the gate is
    consulted first and the pair is read only when it reports nothing.
    """

    @pytest.fixture
    def local_dataset(self, spec: TrainSpec) -> TrainSpec:
        """A readable local ``meta/info.json``, so the pair check is reachable."""
        meta = pathlib.Path(spec.dataset_root) / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text('{"total_episodes": 10, "total_tasks": 1}', encoding="utf-8")
        return spec

    def test_an_opt_out_beside_a_split_is_refused_as_the_flag(self, local_dataset: TrainSpec) -> None:
        _set(local_dataset, streaming="false", val_episodes=2)
        problems = LerobotTrainer().validate(local_dataset)
        assert any("streaming must be a boolean" in p for p in problems), problems
        # Measured on d2f23d8: the pair refusal, whose remedy the caller had
        # already followed.
        assert not any("set streaming=False" in p for p in problems), problems

    def test_a_real_opt_in_beside_a_split_is_still_the_pair_refusal(self, local_dataset: TrainSpec) -> None:
        """Control: the guard narrows nothing the pair check already refused."""
        local_dataset.streaming = True
        local_dataset.val_episodes = 2
        problems = LerobotTrainer().validate(local_dataset)
        assert any("set streaming=False" in p for p in problems), problems
        assert not any("must be a boolean" in p for p in problems), problems


class TestTheProviderAgnosticToolReportsTheRefusal:
    """``train_policy`` reaches the gate through ``validate`` on every spec action."""

    @pytest.mark.parametrize("action", ["validate", "train", "export"])
    @pytest.mark.parametrize("field", ["resume", "streaming"])
    def test_a_spec_action_is_refused_by_name(self, tmp_path: pathlib.Path, action: str, field: str) -> None:
        envelope = _run_tool(
            action=action,
            provider="lerobot_local",
            dataset_root=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            **{field: "false"},
        )
        assert envelope["status"] == "error"
        assert f"{field} must be a boolean" in _text(envelope), _text(envelope)

    def test_an_action_that_builds_no_spec_is_not_refused_for_one(self) -> None:
        envelope = _run_tool(action="list", resume="false", streaming="no")
        assert envelope["status"] == "success"
        assert "must be a boolean" not in _text(envelope)
