"""Microduck weight 404 must name the nearest shipped weight(s).

Compare: strands_robots/policies/factory.py:399 does the same difflib
.get_close_matches(..., cutoff=0.6) dance when create_policy() is called with
an unknown provider name. Nine Microduck weights ship on the Hub, so a typo
in `onnx_path` is the exact case the hint was invented for.
"""
from __future__ import annotations

import pytest

from strands_robots.policies.microduck.policy import (
    _MICRODUCK_SHIPPED_WEIGHTS,
    resolve_microduck_weight,
)


def test_404_names_nearest_shipped_weight() -> None:
    """A one-letter typo must list the real neighbour in the refusal."""
    with pytest.raises(FileNotFoundError) as info:
        resolve_microduck_weight("alpha_walkinng.onnx")  # one extra 'n'
    msg = str(info.value)
    assert "Did you mean" in msg, msg
    assert "alpha_walking.onnx" in msg, msg


def test_404_hint_is_belt_and_braces() -> None:
    """A hint must come even if the Hub listing is unreachable - the shipped
    weight tuple exists for exactly this fallback. We can't easily hide the
    Hub in-process, so this test only pins the fallback list stays non-empty
    and contains the names documented at docs/learn/policies/microduck.md.
    """
    assert "alpha_walking.onnx" in _MICRODUCK_SHIPPED_WEIGHTS
    assert "alpha_stand.onnx" in _MICRODUCK_SHIPPED_WEIGHTS
    assert len(_MICRODUCK_SHIPPED_WEIGHTS) >= 9  # the family at v0.5.3
