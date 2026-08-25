"""``lerobot_train`` refuses a checkpoint cadence lerobot cannot decode.

``build_train_command`` writes ``save_freq`` into ``--save_freq=`` - and, when a
validation split is requested, into ``--eval_steps=`` as well - in the argv of a
process the tool then launches DETACHED. So one unusable value spoils two flags
and neither refusal reaches the caller: the tool returns a pid and a log path,
and only the training log, minutes later, records that lerobot could not parse
the token.

Every knob beside it in that argv was already checked up front for exactly this
reason - ``steps`` and ``batch_size`` against the shared positive-count domain,
``num_gpus``, ``val_episodes``, ``lora_r`` / ``lora_alpha``, and ``device``
against torch's own device grammar. ``save_freq`` was the last one carried
through unchecked, and it failed in two ways:

* Written into the argv as-is, so ``True``, ``2.7``, ``5000.0``, ``nan`` and
  ``inf`` all reached lerobot's ``int`` field, which decodes the token with
  ``int(...)`` and refuses every one of them - inside the detached process.
  A caller who asked to checkpoint every ``2.7`` steps was told the run started.
* Compared against zero to pick the ``--eval_steps`` fallback
  (``save_freq if save_freq > 0 else steps``), so a string or a list raised
  ``TypeError`` out of the builder itself, surfacing through the tool's outermost
  handler as ``Tool execution failed: '>' not supported between instances of
  'str' and 'int'`` - a message naming neither the parameter nor the tool.

The floor is deliberately NOT part of this domain, and these tests pin that too:
lerobot documents a non-positive ``save_freq`` as "disables periodic saving" and
``should_save_checkpoint`` implements it, so ``0`` and a negative are a
capability. Only the type is graded, which is why the guard sits beside
``device`` - after the resume early return, so a resumed run whose argv carries
neither flag is never refused for a value it does not use.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod
from tests.tools.test_lerobot_train import _FakeProc, _write_dataset

build_train_command = train_mod.build_train_command

# Cadences lerobot cannot decode from the argv token, and why each one matters.
UNDECODABLE_CADENCES: tuple[Any, ...] = (
    True,  # int subclass: renders as the token "True"
    False,  # likewise, and it is not the disable sentinel either
    2.7,  # fractional: the int field cannot decode it
    5000.0,  # integral but not an int - the same decoding failure
    float("nan"),
    float("inf"),
    "5000",  # raised a bare TypeError out of the eval_steps comparison
    [5000],  # likewise
    object(),  # a value with no numeric meaning at all
)

# The two documented spellings of "write only the final checkpoint".
DISABLING_CADENCES: tuple[int, ...] = (0, -5)


def _build(**kwargs: Any) -> list[str]:
    """Build an argv with a minimal usable base, overridden by ``kwargs``.

    Funnelled so the deliberately off-type values below reach the runtime guard
    the way an agent supplies them, without a type checker objecting at each
    call site.
    """
    base: dict[str, Any] = {"dataset_root": "/data/cubes", "policy_type": "act"}
    base.update(kwargs)
    return build_train_command(**base)


def _call_tool(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool on a ``start`` action, overridden by ``kwargs``.

    Funnelled for the same reason as :func:`_build`: the values under test are
    deliberately off-type, and they have to reach the runtime guard the way an
    agent supplies them.
    """
    base: dict[str, Any] = {"action": "start"}
    base.update(kwargs)
    return train_mod.lerobot_train(**base)


def _flag(argv: list[str], name: str) -> str | None:
    """The value of ``--name=`` in *argv*, or ``None`` when it is absent."""
    prefix = f"--{name}="
    for token in argv:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _token_decodes(value: Any) -> bool:
    """Can lerobot's parser read the argv token *value* renders to?

    ``--save_freq=`` carries ``f"{value}"``, and lerobot declares the field as a
    plain ``int``, so this is the first of the two consumers a cadence has to
    survive.
    """
    import draccus

    try:
        draccus.decode(int, f"{value}")
    except Exception:  # noqa: BLE001 - any decoding failure means "not readable"
        return False
    return True


def _comparison_survives(value: Any) -> bool:
    """Does ``value > 0`` - the ``--eval_steps`` fallback selector - not raise?

    The second consumer, and the one that produced a bare ``TypeError`` from
    inside the builder for a value whose *token* would have decoded.
    """
    try:
        _ = value > 0
    except TypeError:
        return False
    return True


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the on-disk session store inside the test's own tmp_path."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class TestACadenceLerobotCannotDecodeIsRefused:
    """The regression: an undecodable cadence must never reach the argv."""

    @pytest.mark.parametrize("value", UNDECODABLE_CADENCES)
    def test_an_undecodable_cadence_never_reaches_the_argv(self, value: Any) -> None:
        with pytest.raises(ValueError, match="save_freq must be an integer number of steps"):
            _build(save_freq=value)

    @pytest.mark.parametrize("value", UNDECODABLE_CADENCES)
    def test_the_refusal_names_the_parameter_and_quotes_the_value(self, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(save_freq=value)
        message = str(excinfo.value)
        assert "lerobot_train: save_freq" in message, message
        assert repr(value) in message, message

    @pytest.mark.parametrize("value", UNDECODABLE_CADENCES)
    def test_the_same_cadence_is_refused_when_a_validation_split_is_requested(self, value: Any, tmp_path: Path) -> None:
        """The path that also spends the value on ``--eval_steps``."""
        dataset = _write_dataset(tmp_path / "cubes")
        with pytest.raises(ValueError, match="save_freq must be an integer number of steps"):
            _build(dataset_root=str(dataset), save_freq=value, val_episodes=2)

    @pytest.mark.parametrize("value", UNDECODABLE_CADENCES)
    def test_no_argv_token_carries_the_undecodable_value(self, value: Any, tmp_path: Path) -> None:
        """Both flags it feeds, not just the one named after it."""
        dataset = _write_dataset(tmp_path / "cubes")
        with pytest.raises(ValueError):
            _build(dataset_root=str(dataset), save_freq=value, val_episodes=2)


class TestAUsableCadenceIsUntouched:
    """The domain must not cost the callers that were already right."""

    @pytest.mark.parametrize("value", [1, 500, 5000, 20_000])
    def test_a_usable_cadence_still_reaches_the_argv(self, value: int) -> None:
        assert _flag(_build(save_freq=value), "save_freq") == str(value)

    def test_none_omits_the_flag_and_keeps_lerobots_own_default(self) -> None:
        assert _flag(_build(save_freq=None), "save_freq") is None

    def test_the_validation_cadence_still_follows_a_usable_save_freq(self, tmp_path: Path) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        argv = _build(dataset_root=str(dataset), save_freq=500, val_episodes=2)
        assert _flag(argv, "eval_steps") == "500"


class TestTheFloorIsNotPartOfThisDomain:
    """A non-positive cadence is a documented capability, not a bad value."""

    @pytest.mark.parametrize("value", DISABLING_CADENCES)
    def test_a_disabling_cadence_still_reaches_the_argv(self, value: int) -> None:
        assert _flag(_build(save_freq=value), "save_freq") == str(value)

    @pytest.mark.parametrize("value", DISABLING_CADENCES)
    def test_a_disabling_cadence_still_falls_back_to_a_single_final_evaluation(
        self, value: int, tmp_path: Path
    ) -> None:
        """The fallback beside the guard is written for exactly this case."""
        dataset = _write_dataset(tmp_path / "cubes")
        argv = _build(dataset_root=str(dataset), save_freq=value, steps=1234, val_episodes=2)
        assert _flag(argv, "eval_steps") == "1234"

    def test_lerobot_really_honors_a_disabling_cadence(self) -> None:
        """Non-vacuity for the scope decision above."""
        lerobot_train_script = pytest.importorskip("lerobot.scripts.lerobot_train")
        should_save = lerobot_train_script.should_save_checkpoint
        assert should_save(50, 0, 100) is False, "no periodic checkpoint"
        assert should_save(100, 0, 100) is True, "but the final one is still written"


class TestTheRefusalReachesTheCallerBeforeAnyProcessStarts:
    """A rejected cadence must be reported, not launched and then discovered."""

    def test_the_tool_reports_an_error_envelope_rather_than_raising(self, tmp_path: Path) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        result = _call_tool(dataset_root=str(dataset), save_freq=2.7)
        assert result["status"] == "error"
        text = "\n".join(item["text"] for item in result["content"] if "text" in item)
        assert "save_freq must be an integer number of steps" in text, text

    def test_a_string_cadence_is_reported_as_a_domain_refusal_not_a_comparison_failure(self, tmp_path: Path) -> None:
        """The value that used to surface a bare ``TypeError`` from ``>``."""
        dataset = _write_dataset(tmp_path / "cubes")
        result = _call_tool(dataset_root=str(dataset), save_freq="5000", val_episodes=2)
        assert result["status"] == "error"
        text = "\n".join(item["text"] for item in result["content"] if "text" in item)
        assert "save_freq must be an integer number of steps" in text, text
        assert "not supported between instances of" not in text, text

    def test_a_refused_cadence_launches_no_process(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        launched: list[Any] = []

        def _fail_if_launched(*args: Any, **kwargs: Any) -> _FakeProc:
            launched.append(args)
            return _FakeProc()

        monkeypatch.setattr(train_mod.subprocess, "Popen", _fail_if_launched)
        result = _call_tool(dataset_root=str(dataset), save_freq=True)
        assert result["status"] == "error"
        assert launched == [], "a refused checkpoint cadence still spawned a training process"


class TestAResumedRunIsNotRefusedForAFlagItDoesNotCarry:
    """The guard sits beside ``device``, after the resume early return."""

    @staticmethod
    def _resumable(root: Path) -> Path:
        checkpoint = root / "checkpoints" / "last" / "pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "train_config.json").write_text(json.dumps({"steps": 10}))
        return root

    def test_a_resumed_run_builds_despite_an_undecodable_cadence(self, tmp_path: Path) -> None:
        output_dir = self._resumable(tmp_path / "out")
        argv = _build(output_dir=str(output_dir), resume=True, save_freq=2.7)
        assert any(token.startswith("--config_path=") for token in argv), argv
        assert _flag(argv, "save_freq") is None, argv

    def test_the_same_cadence_is_refused_for_a_fresh_run(self, tmp_path: Path) -> None:
        """Non-vacuity: the resume path is what excuses the value, not the guard."""
        with pytest.raises(ValueError, match="save_freq must be an integer number of steps"):
            _build(output_dir=str(tmp_path / "out"), resume=True, save_freq=2.7)


class TestThePremisesTheDomainRestsOn:
    """Executable premises, so the reasoning cannot silently become wrong."""

    def test_lerobot_declares_the_cadence_as_a_plain_int_field(self) -> None:
        train_config = pytest.importorskip("lerobot.configs.train")
        declared = {field.name: field.type for field in dataclasses.fields(train_config.TrainPipelineConfig)}
        assert declared["save_freq"] is int

    @pytest.mark.parametrize("value", UNDECODABLE_CADENCES)
    def test_every_refused_cadence_fails_at_least_one_consumer(self, value: Any) -> None:
        """Nothing is refused that both consumers could have honored."""
        pytest.importorskip("draccus")
        assert not (_token_decodes(value) and _comparison_survives(value)), (
            f"{value!r} is honored by lerobot's parser AND by the eval_steps comparison"
        )

    def test_both_failure_modes_are_really_represented(self) -> None:
        """Non-vacuity: neither consumer alone accounts for the probe set.

        A guard argued from the argv token alone would have missed ``'5000'`` -
        it renders to the very token an int field decodes - and one argued from
        the comparison alone would have missed every float and bool, which
        compare against zero without complaint.
        """
        pytest.importorskip("draccus")
        token_only = [v for v in UNDECODABLE_CADENCES if not _token_decodes(v) and _comparison_survives(v)]
        comparison_only = [v for v in UNDECODABLE_CADENCES if _token_decodes(v) and not _comparison_survives(v)]
        assert token_only, "no value fails only lerobot's parser"
        assert comparison_only, "no value fails only the eval_steps comparison"

    @pytest.mark.parametrize("value", DISABLING_CADENCES)
    def test_a_disabling_cadence_is_honored_by_both_consumers(self, value: int) -> None:
        """The other half: the values kept out of scope really are usable."""
        pytest.importorskip("draccus")
        assert _token_decodes(value) and _comparison_survives(value)

    def test_the_run_size_knobs_beside_it_hold_the_same_int_requirement(self) -> None:
        """The domain is the sibling's, with the floor removed - not a new one."""
        from strands_robots.utils import positive_count_error

        assert positive_count_error(5000.0, "steps", "lerobot_train") is not None
        assert positive_count_error(5000, "steps", "lerobot_train") is None

    def test_the_comparison_that_used_to_raise_is_the_eval_steps_fallback(self) -> None:
        """Grounds the second failure mode in the source that produced it."""
        import inspect

        source = inspect.getsource(train_mod.build_train_command)
        assert "save_freq if save_freq > 0 else steps" in source


def test_the_probe_set_covers_every_way_the_token_fails() -> None:
    """Guards the probe set itself against a future edit dropping a case."""
    floats = [value for value in UNDECODABLE_CADENCES if isinstance(value, float)]
    assert any(math.isnan(value) for value in floats)
    assert any(math.isinf(value) for value in floats)
    assert any(not float(value).is_integer() for value in floats), "a fractional cadence"
    assert any(float(value).is_integer() for value in floats), "an integral non-int cadence"
    assert any(isinstance(value, bool) for value in UNDECODABLE_CADENCES)
    assert any(isinstance(value, str) for value in UNDECODABLE_CADENCES)
    plain_ints = [v for v in UNDECODABLE_CADENCES if isinstance(v, int) and not isinstance(v, bool)]
    assert plain_ints == [], f"a plain int belongs to the accepted domain, not the probe set: {plain_ints}"
