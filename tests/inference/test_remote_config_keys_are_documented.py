# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One page documents every knob the ``remote`` provider accepts.

``strands_robots/registry/policies.json`` declares the ``config_keys`` a caller
may pass through ``create_policy("remote", ...)``, and ``docs/inference/remote.md``
is the page every other surface sends that caller to -- the provider matrix in
``docs/policies/overview.md``, the ``strands_robots.inference`` package docstring
and the nav row all name it.

Until this file's companion change ``connect_timeout`` and ``request_timeout``
were documented on a second, 316-word page instead, so the page a reader is sent
to named three of the five keys and neither read deadline. A knob that is absent
from the page it is looked up on is not discoverable, and a default that is only
stated in prose drifts silently when the constructor changes: both rules are
graded here from the registry and from
:class:`~strands_robots.inference.RemotePolicy`'s own signature, so the
expectation tracks the code rather than a second copy of it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from strands_robots.inference import RemotePolicy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC = _REPO_ROOT / "docs" / "inference" / "remote.md"
_REGISTRY = _REPO_ROOT / "strands_robots" / "registry" / "policies.json"

#: A registry entry that had shrunk to a key or two would make every rule below
#: pass without proving anything.
_MINIMUM_KEYS = 5


def _config_keys() -> list[str]:
    """The keys the shipped registry says ``remote`` accepts."""
    providers = json.loads(_REGISTRY.read_text(encoding="utf-8"))["providers"]
    return list(providers["remote"]["config_keys"])


def _page() -> str:
    return _DOC.read_text(encoding="utf-8")


def _documented_defaults() -> list[tuple[str, str]]:
    """Every config key whose constructor default is a value a page can spell."""
    params = inspect.signature(RemotePolicy.__init__).parameters
    pairs: list[tuple[str, str]] = []
    for key in _config_keys():
        default = params[key].default if key in params else inspect.Parameter.empty
        if isinstance(default, bool) or not isinstance(default, int | float | str):
            continue  # ``endpoint`` has no value to state
        pairs.append((key, str(default)))
    return pairs


def test_the_registry_entry_is_big_enough_for_this_to_mean_anything() -> None:
    """Keeps the two rules below from passing vacuously."""
    assert len(_config_keys()) >= _MINIMUM_KEYS
    assert _documented_defaults(), "no config key resolves to a constructor default"


@pytest.mark.parametrize("key", _config_keys())
def test_every_config_key_is_named_on_the_page(key: str) -> None:
    """A knob a caller may pass is named on the page the caller is sent to.

    Either spelling counts: the page names some keys as the kwarg it passes
    (``host=``) and others as the config key itself (``connect_timeout``).
    """
    assert f"`{key}`" in _page() or f"`{key}=`" in _page(), (
        f"docs/inference/remote.md does not name the `{key}` config key the remote "
        "provider accepts - a reader sent to this page cannot look the knob up"
    )


@pytest.mark.parametrize(("key", "default"), _documented_defaults())
def test_the_documented_default_is_the_constructor_default(key: str, default: str) -> None:
    """The stated default is RemotePolicy's, so changing one side reds here."""
    assert default in _page(), (
        f"RemotePolicy defaults {key} to {default}, which docs/inference/remote.md "
        "does not state - the page and the constructor have drifted"
    )
