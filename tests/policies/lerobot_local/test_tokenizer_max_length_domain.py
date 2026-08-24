"""The instruction token budget must be a count the tokenizer can slice by.

``LerobotLocalPolicy`` hands ``tokenizer_max_length`` to a HuggingFace tokenizer
as ``max_length`` alongside ``truncation=True`` and ``padding="max_length"``, so
the tokenizer reads it as a slice bound over the encoded instruction. Four
sibling parameters of the same constructor are already refused on arrival
(``actions_per_step`` and ``rtc_execution_horizon`` through ``chunk_count_error``,
``image_keys`` and ``robot_state_keys`` through ``name_list_error``), and the
docstring of ``rtc_execution_horizon`` states the reason: it bounds a slice, so
only a positive ``int`` can be honored and anything else is refused where the
caller names it.

The token budget was the one count in that signature stored verbatim, and it is
the one whose out-of-domain values are hardest to notice. Measured against a real
HuggingFace tokenizer, an 11-token instruction encodes as:

* ``48`` (the default) - 48 wide, 11 attended, the full instruction.
* ``0`` and ``False`` - shape ``(1, 0)``, zero attended tokens, the decoded
  prompt is ``''``. A language-conditioned VLA is asked to act with its entire
  task specification removed, and nothing on any path reports it.
* ``True`` - one token: the instruction's first word only.
* ``None`` - the tokenizer's own ``model_max_length``, a 262144-wide tensor per
  inference step instead of 48.
* ``-5`` - ``OverflowError``; ``2.7`` / ``nan`` / ``inf`` / ``"48"`` / ``[48]`` -
  ``TypeError`` out of the tokenizer's binding, naming neither this parameter nor
  the policy, and only once inference has begun.

These tests pin the domain at both surfaces the siblings guard, that the refusal
precedes the checkpoint download, that the two surfaces cannot diverge, and -
with a tokenizer that models the slice-bound contract this provider relies on -
that the refused counts are exactly the ones that lose the instruction.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.utils import positive_count_error

#: Counts a tokenizer can slice an instruction by.
USABLE: list[Any] = [1, 8, 48, 512]

#: Values no tokenizer can take as a slice bound over the instruction. ``0`` and
#: ``False`` empty the prompt, ``True`` keeps one token, ``None`` falls back to
#: the tokenizer's ``model_max_length``, and the rest reach its binding as a
#: type it cannot interpret as an integer.
UNUSABLE: list[Any] = [0, -5, True, False, 2.7, 48.0, math.nan, math.inf, -math.inf, "48", None, [48]]


def _construct(_loads: list[bool] | None = None, **overrides: Any) -> LerobotLocalPolicy:
    """Build the policy without fetching a checkpoint.

    ``__init__`` calls ``_load_model`` last, so a guard that raises before it
    provably costs no download. ``_loads`` collects one entry per call, which is
    what lets a test assert the refusal came first.
    """

    def _record(_self: LerobotLocalPolicy) -> None:
        if _loads is not None:
            _loads.append(True)

    with patch.object(LerobotLocalPolicy, "_load_model", _record):
        return LerobotLocalPolicy(pretrained_name_or_path="test/model", **overrides)


def _construction_refusal(value: Any) -> str | None:
    """The refusal text ``__init__`` reports for ``value``, or ``None``."""
    try:
        _construct(tokenizer_max_length=value)
    except ValueError as exc:
        return str(exc)
    return None


def _preflight_refusal(value: Any) -> str | None:
    """The refusal text ``preflight`` reports for ``value``, or ``None``.

    ``preflight`` takes the configuration as keywords, so the value is named the
    way a caller names it rather than wrapped in a dict.
    """
    try:
        LerobotLocalPolicy.preflight(set(), pretrained_name_or_path="test/model", tokenizer_max_length=value)
    except ValueError as exc:
        return str(exc)
    return None


class TestAnUnusableTokenBudgetIsRefusedAtConstruction:
    """The value is refused where the caller names it, before any download."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_construction_refuses_it(self, value: Any) -> None:
        refusal = _construction_refusal(value)
        assert refusal is not None, f"tokenizer_max_length={value!r} was accepted"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_names_the_parameter_and_the_provider(self, value: Any) -> None:
        refusal = _construction_refusal(value)
        assert refusal is not None
        assert "tokenizer_max_length" in refusal
        assert refusal.startswith("lerobot_local:")

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_refusal_precedes_the_checkpoint_load(self, value: Any) -> None:
        """No weights are fetched for a value the provider cannot honor."""
        loads: list[bool] = []
        with pytest.raises(ValueError):
            _construct(loads, tokenizer_max_length=value)
        assert loads == [], "the refused construction reached the checkpoint load"

    def test_a_usable_budget_does_reach_the_checkpoint_load(self) -> None:
        """The ordering assertion above is not vacuous: the load is reachable."""
        loads: list[bool] = []
        _construct(loads, tokenizer_max_length=48)
        assert loads == [True]


class TestAUsableTokenBudgetIsAccepted:
    """The guard refuses only what no tokenizer can slice by."""

    @pytest.mark.parametrize("value", USABLE)
    def test_construction_accepts_it(self, value: Any) -> None:
        assert _construction_refusal(value) is None

    @pytest.mark.parametrize("value", USABLE)
    def test_the_accepted_value_is_stored_verbatim(self, value: Any) -> None:
        policy = _construct(tokenizer_max_length=value)
        assert policy._tokenizer_max_length == value

    def test_the_default_is_accepted(self) -> None:
        policy = _construct()
        assert policy._tokenizer_max_length == 48


class TestPreflightRefusesTheSameBudgets:
    """The rollout entry point reports it before the weights are fetched."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_preflight_refuses_it(self, value: Any) -> None:
        refusal = _preflight_refusal(value)
        assert refusal is not None, f"preflight accepted tokenizer_max_length={value!r}"
        assert "tokenizer_max_length" in refusal

    @pytest.mark.parametrize("value", USABLE)
    def test_preflight_accepts_a_usable_budget(self, value: Any) -> None:
        assert _preflight_refusal(value) is None

    def test_preflight_says_nothing_when_the_key_is_absent(self) -> None:
        """Omitting it asks for the provider default, which is not a caller value."""
        LerobotLocalPolicy.preflight(set(), pretrained_name_or_path="test/model")


class TestTheTwoSurfacesAgree:
    """One parameter, one accepted domain, whichever surface receives it."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_construction_and_preflight_reach_the_same_verdict(self, value: Any) -> None:
        assert (_construction_refusal(value) is None) == (_preflight_refusal(value) is None)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_both_surfaces_report_the_same_text(self, value: Any) -> None:
        assert _construction_refusal(value) == _preflight_refusal(value)


class TestTheDomainIsTheSharedOne:
    """The provider adds nothing to the shared count rule, so it cannot drift."""

    @pytest.mark.parametrize("value", [*USABLE, *UNUSABLE])
    def test_the_verdict_matches_the_shared_domain(self, value: Any) -> None:
        shared = positive_count_error(value, "tokenizer_max_length", "lerobot_local")
        assert _construction_refusal(value) == shared

    def test_an_integral_float_is_refused_because_the_tokenizer_cannot_take_one(self) -> None:
        """``48.0`` is the boundary case the wider whole-number domain would admit.

        The tokenizer's binding refuses any ``float`` outright ("``'float' object
        cannot be interpreted as an integer``"), so the strict-``int`` domain is
        the one that matches the consumer.
        """
        assert _construction_refusal(48.0) is not None
        assert _construction_refusal(48) is None


class _SlicingTokenizer:
    """A tokenizer modelling the slice-bound contract this provider relies on.

    HuggingFace tokenizers take ``max_length`` as a bound on the encoded
    sequence and, with ``truncation=True``, cut to it. This reproduces that so
    the consequence of an out-of-domain count is executable without a download;
    the real behaviour is quoted in the module docstring.
    """

    model_max_length = 262144
    padding_side = "right"

    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids = [hash(word) % 1000 + 1 for word in text.split()]
        limit = kwargs.get("max_length")
        if limit is None:
            limit = self.model_max_length
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError(f"argument 'max_length': {type(limit).__name__} object cannot be interpreted as an integer")
        if limit < 0:
            raise OverflowError("can't convert negative int to unsigned")
        kept = ids[:limit]
        mask = [1] * len(kept) + [0] * (limit - len(kept))
        kept = kept + [self.pad_token_id] * (limit - len(kept))
        return {"input_ids": torch.tensor([kept]), "attention_mask": torch.tensor([mask])}


class TestAnUnusableBudgetLosesTheInstruction:
    """Why the domain refuses these counts, measured through the real code path."""

    INSTRUCTION = "pick up the red block and place it in the bowl"

    def _tokenize(self, budget: Any) -> tuple[int, int]:
        """Return ``(width, attended)`` for ``budget`` via ``_tokenize_instruction``."""
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy._tokenizer = _SlicingTokenizer()
        policy._tokenizer_max_length = budget
        policy._tokenizer_padding_side = "right"
        policy._device = torch.device("cpu")
        policy._policy = None
        result = policy._tokenize_instruction(self.INSTRUCTION)
        assert result is not None
        tokens, mask = result
        assert mask is not None
        return tokens.shape[1], int(mask.sum())

    def test_a_usable_budget_keeps_the_whole_instruction(self) -> None:
        width, attended = self._tokenize(48)
        assert width == 48
        assert attended == len(self.INSTRUCTION.split())

    def test_zero_removes_the_instruction_entirely(self) -> None:
        """The case the guard exists for: an empty prompt, reported nowhere."""
        assert self._tokenize(0) == (0, 0)
        assert _construction_refusal(0) is not None

    def test_a_boolean_keeps_only_the_first_token(self) -> None:
        assert self._tokenize(1) == (1, 1)
        assert _construction_refusal(True) is not None

    def test_none_falls_back_to_the_tokenizers_own_ceiling(self) -> None:
        width, attended = self._tokenize(None)
        assert width == _SlicingTokenizer.model_max_length
        assert attended == len(self.INSTRUCTION.split())
        assert _construction_refusal(None) is not None
