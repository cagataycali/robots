"""The submit form's vocabulary must cover train_policy's, or say why it does not.

SPEC_KEYS is the set of fields the dashboard will forward to train_policy; anything else is
refused BY NAME (Q6). That makes it a whitelist, and a whitelist silently rots: `val_episodes`
had been absent for as long as the field existed, so a policy trained from this dashboard could
never hold out a validation set -- the operator got a loss curve that cannot distinguish learning
from memorising, and no message ever mentioned it. Asking a client for it answered:

    unknown field(s): val_episodes. Valid fields: provider, dataset_root, ...

These tests make the two lists grade each other: every train_policy parameter is either sendable
from the form or listed in _NOT_IN_FORM with a reason.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.dashboard import training
from strands_robots.tools.train_policy import train_policy

_PARAMS = set(inspect.signature(train_policy).parameters)


def test_every_train_policy_parameter_is_either_sendable_or_explained() -> None:
    accounted = set(training.SPEC_KEYS) | set(training._NOT_IN_FORM)
    missing = sorted(_PARAMS - accounted)
    assert not missing, (
        f"train_policy accepts {missing} but the form neither sends nor explains them. Add each to "
        "SPEC_KEYS (if an operator can answer it here) or to _NOT_IN_FORM with the reason -- an "
        "unlisted field is refused by name, which reads to the user like a bug in their request."
    )


def test_the_form_never_offers_a_field_train_policy_would_reject() -> None:
    """The other direction: a SPEC_KEY that is not a real parameter becomes a TypeError at submit."""
    invented = sorted(set(training.SPEC_KEYS) - _PARAMS)
    assert not invented, f"SPEC_KEYS names {invented}, which train_policy does not accept"


def test_no_key_is_in_both_lists() -> None:
    both = sorted(set(training.SPEC_KEYS) & set(training._NOT_IN_FORM))
    assert not both, f"{both} is both sent and explained-away; one of the two is stale"


def test_every_exemption_states_a_reason() -> None:
    for key, reason in training._NOT_IN_FORM.items():
        assert len(reason.split()) >= 4, f"{key}'s exemption is not a reason: {reason!r}"


def test_val_episodes_is_accepted_and_reaches_the_spec() -> None:
    """The field that started this: it must pass the body check and arrive as a kwarg."""
    seen: dict[str, object] = {}

    def fake_train_policy(**kwargs):
        seen.update(kwargs)
        return {"status": "success", "content": [{"text": "ok"}]}

    import strands_robots.tools.train_policy as tp

    original = tp.train_policy
    tp.train_policy = fake_train_policy  # type: ignore[assignment]
    try:
        training.submit(
            {
                "provider": "mock",
                "dataset_root": "/tmp/x",
                "output_dir": "/tmp/o",
                "steps": 10,
                "val_episodes": 2,
            }
        )
    finally:
        tp.train_policy = original

    assert seen.get("val_episodes") == 2, seen


def test_a_holdout_adds_no_complaint_of_its_own() -> None:
    """val_episodes must not make validate() refuse something it otherwise accepts.

    The bound is the dataset's episode count from meta/info.json, so a bad value is the trainer's
    to refuse (_validation_episodes_problems) -- this pins that a *good* value is silent, which is
    what makes the new field safe to offer.
    """
    base = {"provider": "mock", "dataset_root": "/tmp/x", "output_dir": "/tmp/o", "steps": 10}
    assert training.validate(base)["text"] == training.validate({**base, "val_episodes": 2})["text"]


def test_the_coverage_test_would_have_caught_the_missing_field(monkeypatch) -> None:
    """Non-vacuity: with val_episodes taken back out of SPEC_KEYS, coverage must FAIL."""
    monkeypatch.setattr(training, "SPEC_KEYS", tuple(k for k in training.SPEC_KEYS if k != "val_episodes"))
    with pytest.raises(AssertionError, match="val_episodes"):
        test_every_train_policy_parameter_is_either_sendable_or_explained()
