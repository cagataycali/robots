"""``allow_missing_models`` is checked at construction, not read by truthiness.

:class:`~strands_robots.policies.wbc.WBCPolicy` takes two posture flags in one
signature. ``walk`` was checked with
:func:`~strands_robots.utils.boolean_flag_error`; ``allow_missing_models`` sat
beside it and was read by ``if not allow_missing_models``. Every non-empty
string is truthy, so ``"false"`` - the spelling a JSON ``policy_config`` reaches
for to ask for the eager load - selected the test seam instead: no session was
loaded, construction succeeded, and the missing checkpoint surfaced at the first
``get_actions`` as a refusal advising ``allow_missing_models=False``, the value
the caller believed they had passed. ``None``, ``0`` and ``[]`` took the loading
branch without ever being a declared spelling of it.

The check lives on the base class only. :class:`~strands_robots.policies.wbc.WBCGaitPolicy`
forwards the flag to ``super().__init__``, so one site covers both providers,
and it precedes the load it gates so a refused value reaches no loader.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBCPolicy
from strands_robots.policies.wbc.gait import WBCGaitPolicy
from strands_robots.utils import boolean_flag_error

#: Every value below is either a boolean or something the shared domain refuses.
#: The refused half is derived from the domain rather than copied, so a spelling
#: the domain later admits or refuses is graded without an edit here.
_CANDIDATES: list[Any] = ["false", "False", "no", "off", "0", "none", "true", 1.5, [1], None, 0, 0.0, []]
REFUSED: list[Any] = [v for v in _CANDIDATES if boolean_flag_error(v, "allow_missing_models", "WBCPolicy")]
assert len(REFUSED) == len(_CANDIDATES), "every candidate is meant to be a non-boolean"


def _ids(values: list[Any]) -> list[str]:
    return [repr(v) for v in values]


def _policy(**kwargs: Any) -> WBCPolicy:
    """Construct with the caller's spelling, typed loosely so the check is what refuses."""
    return WBCPolicy(**kwargs)


def _gait_policy(**kwargs: Any) -> WBCGaitPolicy:
    return WBCGaitPolicy(**kwargs)


@pytest.mark.parametrize("value", REFUSED, ids=_ids(REFUSED))
def test_a_non_boolean_is_refused_naming_the_parameter(value: Any) -> None:
    """Neither posture is taken for a value that spells neither."""
    with pytest.raises(ValueError, match="allow_missing_models"):
        _policy(allow_missing_models=value)


def test_the_refusal_is_the_shared_domain_verbatim() -> None:
    """One owner for the wording, so this flag and ``walk`` cannot drift apart."""
    expected = boolean_flag_error("false", "allow_missing_models", "WBCPolicy")
    assert expected is not None
    with pytest.raises(ValueError) as excinfo:
        _policy(allow_missing_models="false")
    assert str(excinfo.value) == expected


def test_the_refusal_precedes_the_load_it_gates() -> None:
    """A falsy non-boolean used to reach the ONNX loader; now it reaches nothing.

    ``0`` is the sharp case: it is falsy, so before the check it took the
    loading branch and the caller read a ``RuntimeError`` about ``onnxruntime``
    or the checkpoint - a diagnosis of the environment for a mistake in the
    argument. The refusal now names the argument, and it does so whether or not
    the extra is installed.
    """
    with pytest.raises(ValueError, match="allow_missing_models"):
        _policy(checkpoint="/nonexistent/checkpoint", allow_missing_models=0)


def test_walk_is_named_first_when_both_flags_are_wrong() -> None:
    """The two checks are ordered; the earlier parameter is the one reported."""
    with pytest.raises(ValueError, match="walk"):
        _policy(walk="false", allow_missing_models="false")


def test_the_gait_variant_forwards_to_the_same_check() -> None:
    """``WBCGaitPolicy`` forwards the flag, so the base class's check covers it."""
    expected = boolean_flag_error("false", "allow_missing_models", "WBCPolicy")
    with pytest.raises(ValueError) as excinfo:
        _gait_policy(allow_missing_models="false")
    assert str(excinfo.value) == expected


@pytest.mark.parametrize("value", [True, np.bool_(True)], ids=["True", "np.bool_(True)"])
def test_the_seam_is_still_honoured_for_a_boolean_true(value: Any) -> None:
    """Over-reach control: the documented seam still skips the eager load."""
    policy = _policy(allow_missing_models=value)
    assert policy.policy_session is None
    assert policy.walk_session is None


@pytest.mark.parametrize("value", [False, np.bool_(False)], ids=["False", "np.bool_(False)"])
def test_a_boolean_false_still_reaches_the_loader(value: Any) -> None:
    """Over-reach control: ``False`` is honoured - it loads, and fails loudly.

    The loader refuses with ``RuntimeError`` here because either
    ``onnxruntime`` is absent or the checkpoint is: both are the documented
    "missing checkpoint fails loudly at construction" behaviour, and neither is
    the argument refusal above.
    """
    with pytest.raises(RuntimeError):
        _policy(checkpoint="/nonexistent/checkpoint", allow_missing_models=value)
