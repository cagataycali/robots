# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every policy-type roster ``docs/training/overview.md`` spells matches its gate.

The Training page's ``validate()`` paragraph tells a reader which policy types a
capability gate accepts - which types normalize with QUANTILES, and which expose
``use_relative_actions``. Those rosters are discovered from lerobot's live
registry at runtime (see
:func:`~strands_robots.training.lerobot._policy_supports_relative_actions` and
its siblings), so the page is a hand-written copy of an answer the code derives,
and a copy no test reads goes stale on the commit that widens the gate: the
relative-action roster was written as ``pi0``/``pi05``/``pi0_fast`` while the
gate had already accepted ``groot``, so the page denied a combination
``validate()`` accepted.

Pinned here: for each gate the page enumerates, the types it names are exactly
the types the gate accepts. The rosters are read from the gates and the page,
neither is spelled in this file, so a lerobot release that adds a policy with
one of these capabilities fails here until the page is updated with it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from strands_robots.training.lerobot import (
    _lerobot_policy_types,
    _policy_supports_relative_actions,
    _policy_uses_quantile_norm,
)

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "training" / "overview.md"

#: Marker of the paragraph that lists what the preflight refuses before launch.
_PARAGRAPH_MARKER = "`validate()` refuses before launch"

#: Each gate the paragraph enumerates: the substring that identifies its clause,
#: and the predicate that answers the same question for one policy type.
_DOCUMENTED_GATES: tuple[tuple[str, str, Callable[[str], bool]], ...] = (
    ("quantile-normalization", "quantiles", _policy_uses_quantile_norm),
    ("relative-actions", "relative_actions", _policy_supports_relative_actions),
)

_BACKTICKED = re.compile(r"`([^`]+)`")


def _refusal_paragraph() -> str:
    """The one paragraph of the page that opens on what ``validate()`` refuses."""
    paragraphs = [p for p in _PAGE.read_text(encoding="utf-8").split("\n\n") if _PARAGRAPH_MARKER in p]
    assert len(paragraphs) == 1, (
        f"expected one '{_PARAGRAPH_MARKER}' paragraph in {_PAGE.name}, found {len(paragraphs)}"
    )
    return paragraphs[0]


def _clause(anchor: str) -> str:
    """The single semicolon-delimited clause of the paragraph naming ``anchor``."""
    clauses = [c for c in _refusal_paragraph().split(";") if anchor in c]
    assert len(clauses) == 1, f"expected one clause naming '{anchor}' in {_PAGE.name}, found {len(clauses)}"
    return clauses[0]


@pytest.mark.parametrize(("gate", "anchor", "probe"), _DOCUMENTED_GATES, ids=[g[0] for g in _DOCUMENTED_GATES])
def test_the_page_names_exactly_the_types_the_gate_accepts(
    gate: str, anchor: str, probe: Callable[[str], bool]
) -> None:
    """A roster on the page is the roster the gate derives - no more, no fewer."""
    known = _lerobot_policy_types()
    assert known, "no LeRobot policy types discovered; the rosters below would be vacuous"

    accepted = {ptype for ptype in known if probe(ptype)}
    assert accepted, f"the {gate} gate accepts no policy type; its documented roster would be vacuous"

    documented = {token for token in _BACKTICKED.findall(_clause(anchor)) if token in known}
    assert documented == accepted, (
        f"the {gate} roster in {_PAGE.name} drifted from the gate: "
        f"page-only={sorted(documented - accepted)} gate-only={sorted(accepted - documented)}"
    )
