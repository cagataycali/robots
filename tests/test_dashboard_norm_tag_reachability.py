"""Can a dashboard-launched run even REQUEST a norm_tag? Today: no. This test says so out loud.

The precheck added in c46dcf2d compares the operator's ``norm_tag`` with the tags a checkpoint's
``norm_stats.json`` declares, so upstream's UnknownNormTagError cannot arrive after the arm is
torqued. Auditing it against the LIVE registry showed the honest limit of that work: ``lerobot_local``
lists ``norm_tag`` in ``unsettable_over_mesh`` and defaults it to None, so the wire schema drops the
field, the run form can never own it, and a mesh-launched run never requests a tag at all. The
precheck is therefore correct but INERT on this path - it protects a direct API caller, not the
operator, and a UI branch for it was removed rather than left implying a control nobody has.

This is a tripwire, not a complaint. The day the registry makes norm_tag settable over the wire, the
precheck becomes operator-facing and the run form has to send it - and that day this test fails and
says exactly that, instead of the capability arriving silently with no screen behind it.
"""
from __future__ import annotations

import pytest

from strands_robots.dashboard import config_api


@pytest.fixture(scope="module")
def lerobot_local() -> dict:
    catalog = config_api._policy_catalog()
    if not catalog:
        pytest.skip("policy registry unavailable in this environment")
    entry = next((p for p in catalog if p.get("name") == "lerobot_local"), None)
    if entry is None:
        pytest.skip("lerobot_local not in this registry")
    return entry


def test_norm_tag_is_still_unsettable_over_the_mesh(lerobot_local: dict) -> None:
    unsettable = list(lerobot_local.get("unsettable_over_mesh") or [])
    wire_keys = [f.get("key") for f in (lerobot_local.get("wire_fields") or [])]
    if "norm_tag" not in unsettable:
        pytest.fail(
            "norm_tag is now settable over the mesh, so a dashboard run CAN request a tag the "
            "checkpoint may not declare. Two things follow: RunForm must send norm_tag with its "
            "policy-fit request (the route already takes it), and the refusal must be readable on "
            "the run form. See tests/test_dashboard_policy_fit_norm_tag.py for the rule itself."
        )
    assert "norm_tag" not in wire_keys, (
        "a field the wire schema drops must not appear in the run form's schema - the form would "
        "collect a value the mesh then silently discards"
    )


def test_the_premise_holds_that_this_provider_declares_the_field_at_all(lerobot_local: dict) -> None:
    # If norm_tag vanished from config_keys entirely, this tripwire is guarding nothing and the
    # precheck can go - which is a decision for a human, so fail loudly rather than skip.
    assert "norm_tag" in list(lerobot_local.get("config_keys") or []), (
        "lerobot_local no longer declares norm_tag at all; re-examine whether the policy_fit "
        "precheck and this tripwire still have a subject"
    )
