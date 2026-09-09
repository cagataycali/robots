"""``lerobot_train`` must refuse a training posture it can only misread.

Each of the five booleans this argv is built from selects a *posture*, not a
magnitude, and each was read by truthiness - so the words a caller reaches for
when opting out selected the affirmative posture, because every non-empty string
is truthy. Measured on ``12dc48d`` with ``policy_type="act"`` unless noted:

* ``resume="false"`` built ``--config_path=<ckpt> --resume=true`` and returned,
  so the fresh run that was asked for became a resume of a previous config and
  ``--policy.device``, ``--steps``, ``--batch_size`` and ``--save_freq`` were all
  silently dropped from the argv;
* ``lora="false"`` emitted ``--peft.method_type=LORA``;
* ``gradient_checkpointing="false"`` on ``pi0`` emitted
  ``--policy.gradient_checkpointing=true``;
* ``push_to_hub="no"`` emitted ``--policy.push_to_hub=no``.

The other three answered with a refusal whose remedy the caller had already
followed, which cannot be acted on at all: ``lora="false",
train_expert_only="false"`` raised "Pick one fine-tuning strategy" at a caller
who picked neither, ``train_expert_only="false"`` raised "only valid for ['pi0',
'pi05', 'smolvla'] policies", and ``gradient_checkpointing="false"`` on ``act``
raised "drop gradient_checkpointing=" at a caller who had spelled exactly that.

Nothing reports the silent postures: the argv goes to a detached process, the
tool answers ``status="success"`` with a pid and a log path, and lerobot parses
each of those argvs without complaint - it is simply told the opposite posture.
That is the reason the run-size numerics, ``device`` and ``save_freq`` in the
same argv are already refused up front
(``tests/tools/test_lerobot_train_size_knob_domain.py`` and siblings), and these
five are the postures beside them that were carried through unchecked.

The flags are held to the shared
:func:`~strands_robots.utils.boolean_flag_error` domain - the one the sibling
builder ``build_lerobot_command`` already applies to its own argv postures
(``tests/tools/test_lerobot_teleoperate_flag_domain.py``) - so a posture is
refused identically wherever it is supplied rather than merely equivalently.

Both surfaces that read the flags refuse them. The builder owns the argv, but is
reached too late for the tool's own reader: ``push_to_hub`` is consulted by the
operator approval gate first, so an opt-out asked a human to approve a Hub
publication nobody requested.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools.lerobot_train import build_train_command

lerobot_train = train_mod.lerobot_train

# The spellings a caller reaches for when opting out (every one truthy), two
# truthy numbers, and the falsy values that are not a declared spelling of the
# negative posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("0", id="str-zero"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(1, id="int-one"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans the shared domain also accepts,
# which arrive from an array-shaped config or a NumPy comparison.
A_BOOLEAN = [
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.True_, id="np-true"),
    pytest.param(np.False_, id="np-false"),
]

FLAGS = ("resume", "lora", "train_expert_only", "gradient_checkpointing", "push_to_hub")


@pytest.fixture
def checkpoint_tree(tmp_path) -> str:
    """An ``output_dir`` holding a resumable checkpoint, so ``resume`` is live.

    Without one, ``resume`` decides nothing and the posture it selects cannot be
    observed in the argv at all.
    """
    config = tmp_path / "out" / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    return str(tmp_path / "out")


def _build(**kwargs: Any) -> list[str]:
    """Call the builder through one funnel.

    The flag values under test are deliberately outside the declared ``bool``,
    which is the point; mypy does not narrow a splatted ``dict[str, Any]``, so
    routing the call through here states that once rather than suppressing it at
    every call site.
    """
    return build_train_command(**{"dataset_root": "/data/ds", **kwargs})


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel, for the same reason as _build."""
    return dict(lerobot_train(**{"dataset_root": "/data/ds", **kwargs}))


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


class TestTheBuilderRefusesAPostureItCanOnlyMisread:
    """Each flag is held to the shared boolean domain, and named when refused."""

    @pytest.mark.parametrize("flag", FLAGS)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_flag_is_refused_by_name(self, flag: str, value: Any, checkpoint_tree: str) -> None:
        with pytest.raises(ValueError, match=rf"\b{flag} must be a boolean"):
            _build(output_dir=checkpoint_tree, policy_type="pi0", **{flag: value})

    @pytest.mark.parametrize("value", A_BOOLEAN)
    @pytest.mark.parametrize("flag", FLAGS)
    def test_a_usable_boolean_is_not_refused(self, flag: str, value: Any, checkpoint_tree: str) -> None:
        assert _build(output_dir=checkpoint_tree, policy_type="pi0", **{flag: value})


class TestTheRefusalNamesTheFlagRatherThanTheEffectItHad:
    """The three misdirected refusals name the flag they were given instead."""

    def test_opting_out_of_both_freezing_strategies_is_not_a_choice_of_both(self) -> None:
        with pytest.raises(ValueError, match=r"\blora must be a boolean") as refusal:
            _build(lora="false", train_expert_only="false")
        assert "mutually exclusive" not in str(refusal.value)

    def test_opting_out_on_a_policy_that_cannot_freeze_names_the_flag(self) -> None:
        with pytest.raises(ValueError, match=r"\btrain_expert_only must be a boolean") as refusal:
            _build(train_expert_only="false", policy_type="act")
        assert "only valid for" not in str(refusal.value)

    def test_the_remedy_offered_is_not_the_one_already_followed(self) -> None:
        with pytest.raises(ValueError, match=r"\bgradient_checkpointing must be a boolean") as refusal:
            _build(gradient_checkpointing="false", policy_type="act")
        assert "drop gradient_checkpointing" not in str(refusal.value)


class TestTheRefusalReplacesTheOppositePosture:
    """The argv the misread posture used to produce is never built."""

    def test_a_fresh_run_does_not_become_a_resume(self, checkpoint_tree: str) -> None:
        with pytest.raises(ValueError, match=r"\bresume must be a boolean"):
            _build(resume="false", output_dir=checkpoint_tree)

    def test_an_adapter_is_not_trained(self) -> None:
        with pytest.raises(ValueError, match=r"\blora must be a boolean"):
            _build(lora="false")

    def test_a_checkpointing_flag_is_not_emitted_as_true(self) -> None:
        with pytest.raises(ValueError, match=r"\bgradient_checkpointing must be a boolean"):
            _build(gradient_checkpointing="false", policy_type="pi0")

    def test_the_publication_token_is_not_a_word_only_lerobot_can_refuse(self) -> None:
        with pytest.raises(ValueError, match=r"\bpush_to_hub must be a boolean"):
            _build(push_to_hub="no")

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (dict(resume=True), "--resume=true"),
            (dict(lora=True), "--peft.method_type=LORA"),
            (dict(gradient_checkpointing=True, policy_type="pi0"), "--policy.gradient_checkpointing=true"),
            (dict(push_to_hub=True), "--policy.push_to_hub=true"),
            (dict(push_to_hub=False), "--policy.push_to_hub=false"),
        ],
    )
    def test_the_posture_a_real_boolean_selects_is_unchanged(
        self, kwargs: dict[str, Any], expected: str, checkpoint_tree: str
    ) -> None:
        assert expected in _build(output_dir=checkpoint_tree, **kwargs)


class TestTheToolRefusesBeforeItActs:
    """The tool refuses ahead of its own readers, and only where they read."""

    @pytest.fixture
    def gate_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Record every approval-gate consultation, and halt if one happens.

        Returning a refusal keeps a reached gate from launching a detached
        training process, so this fixture cannot start one on the pre-fix code it
        is written to fail against.
        """
        calls: list[dict[str, Any]] = []

        def _spy(extra_flags: dict[str, Any], tool_context: Any) -> dict[str, Any]:
            calls.append(dict(extra_flags))
            return {"status": "error", "content": [{"text": "the approval gate was consulted"}]}

        monkeypatch.setattr(train_mod, "_gate_extra_flags", _spy)
        return calls

    @pytest.fixture
    def recorded_dataset(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
        """A dataset the ``start`` preflight accepts, so the gate is reachable.

        Without it the preflight answers "Dataset metadata not found" first and
        the gate is never consulted - which would make the cell below pass on the
        pre-fix code for a reason that has nothing to do with the flag.
        """
        (tmp_path / "ds" / "meta").mkdir(parents=True)
        (tmp_path / "ds" / "meta" / "info.json").write_text('{"total_episodes": 3}', encoding="utf-8")
        monkeypatch.setattr(train_mod, "SESSION_DIR", tmp_path / ".sessions")
        (tmp_path / ".sessions").mkdir()
        return str(tmp_path / "ds")

    def test_an_opt_out_does_not_ask_a_human_to_approve_a_publication(
        self, gate_calls: list[dict[str, Any]], recorded_dataset: str, tmp_path
    ) -> None:
        envelope = _run_tool(
            dataset_root=recorded_dataset,
            push_to_hub="false",
            output_dir=str(tmp_path / "out"),
            session_name="flag_domain",
        )
        assert envelope["status"] == "error"
        assert "push_to_hub must be a boolean" in _text(envelope)
        # Measured on 12dc48d: [{"policy.push_to_hub": "false"}] - an operator
        # asked to approve the publication the caller had opted out of.
        assert gate_calls == []

    @pytest.mark.parametrize("flag", FLAGS)
    def test_the_refusal_precedes_the_lerobot_and_dataset_preflight(self, flag: str) -> None:
        envelope = _run_tool(dataset_root="/nonexistent/dataset", **{flag: "false"})
        assert envelope["status"] == "error"
        assert f"{flag} must be a boolean" in _text(envelope)

    @pytest.mark.parametrize("action", ["list", "status"])
    def test_an_action_that_reads_no_flag_is_not_refused_for_one(
        self, action: str, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(train_mod, "SESSION_DIR", tmp_path / ".sessions")
        (tmp_path / ".sessions").mkdir()
        envelope = _run_tool(action=action, session_name="absent", resume="false")
        assert "must be a boolean" not in _text(envelope)


class TestEveryBooleanParameterHasADomain:
    """A flag added to the builder later cannot skip the domain unnoticed."""

    def test_the_roster_is_every_boolean_the_builder_declares(self) -> None:
        declared = {
            name
            for name, param in inspect.signature(build_train_command).parameters.items()
            if param.annotation in (bool, "bool")
        }
        assert declared, "the builder declares no bool parameter; this rule has stopped reading it"
        assert declared == set(train_mod._ARGV_FLAGS)
