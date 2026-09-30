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


def test_a_provider_already_removed_refuses_instead_of_warning() -> None:
    """The cut itself is a refusal with the replacement in the sentence, not a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="removed in 1.0") as excinfo:
            create_policy("groot", port=5555)
    assert "lerobot_local(policy_type='groot')" in str(excinfo.value)
    assert not (_PAGES / "groot.md").exists(), "a removed provider has no page to warn on"
