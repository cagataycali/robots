"""A provider announced for removal warns at ``create_policy`` and names what replaces it.

#3818 removes nothing without announcing it one minor ahead with the
replacement named. The announcement is the warning, and the provider's page
carries the same notice, so both surfaces are graded here.
"""

import contextlib
import warnings
from pathlib import Path

import pytest

import strands_robots
from strands_robots.policies.factory import _REMOVED_IN_0_7, create_policy
from strands_robots.registry import list_policy_providers

_PAGES = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "learn" / "policies"


@pytest.mark.parametrize("provider", sorted(_REMOVED_IN_0_7))
def test_create_policy_warns_with_the_replacement(provider: str) -> None:
    assert provider in list_policy_providers(), f"{provider!r} is announced but no longer registered"
    with pytest.warns(DeprecationWarning, match="removed in 0.7") as record:
        with contextlib.suppress(Exception):  # construction may need a GPU or a sidecar; the notice may not
            create_policy(provider)
    assert _REMOVED_IN_0_7[provider] in str(record[0].message)
    assert "removed in 0.7" in (_PAGES / f"{provider}.md").read_text().lower()


def test_a_kept_provider_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create_policy("mock")


def test_groot_local_mode_warns_and_service_mode_does_not() -> None:
    from strands_robots.policies.groot.policy import Gr00tPolicy

    with pytest.warns(DeprecationWarning, match=r"model_path=.*removed in 0\.7"):
        with contextlib.suppress(Exception):  # no Isaac-GR00T here; the notice precedes the load
            Gr00tPolicy(model_path="nvidia/GR00T-N1.6-3B")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        Gr00tPolicy(port=5555)
    assert "removed in 0.7" in (_PAGES / "groot.md").read_text()
