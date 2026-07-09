"""The predicates module docstring is the discovery surface for the benchmark DSL.

:mod:`strands_robots.simulation.predicates` is a *closed* registry: the
YAML/JSON benchmark-spec loader refuses any predicate whose name is not in
:data:`PREDICATE_REGISTRY`, so a human- or LLM-authored spec can only use terms
the author can discover. The sole place those terms are enumerated for a reader
is the module docstring's "Available predicates (bool):" and "Available reward
terms (float):" lists - there is no separate docs page. If a registered
predicate is missing from that list it is effectively invisible to spec authors
(the exact way the four base-locomotion reward terms + ``staged_reward`` were
undiscoverable while listed nowhere).

These tests pin the docstring lists to the registry (grouped by
:func:`predicate_kind`) so the discovery surface can never drift out of sync
with the code again: adding a predicate without documenting it - or listing a
name that no longer exists - fails here.
"""

from __future__ import annotations

import re

from strands_robots.simulation import predicates as P


def _doc_names(section_header: str) -> set[str]:
    """Extract the predicate names listed under a docstring section header.

    A section is the header line followed by 4-space-indented ``name(...)``
    entries; it ends at the first non-indented, non-blank line (a new header or
    paragraph). Only the leading identifier of each entry is collected.
    """
    doc = P.__doc__ or ""
    names: set[str] = set()
    capturing = False
    for line in doc.splitlines():
        if line.strip() == section_header:
            capturing = True
            continue
        if not capturing:
            continue
        if not line.strip():
            # blank line (e.g. between the header and the first entry) - keep going
            continue
        m = re.match(r"\s{4,}([a-z][a-z0-9_]*)\s*\(", line)
        if m:
            names.add(m.group(1))
        else:
            # first non-entry line ends the section
            break
    return names


def _registry_names_of_kind(kind: str) -> set[str]:
    return {n for n in P.PREDICATE_REGISTRY if P.predicate_kind(n) == kind}


class TestDocstringDiscoverySurfaceExists:
    """The docstring must actually carry both enumerated lists."""

    def test_both_section_headers_present(self):
        doc = P.__doc__ or ""
        assert "Available predicates (bool):" in doc
        assert "Available reward terms (float):" in doc

    def test_registry_is_fully_classified(self):
        # Every built-in registry entry must be bool or float so the two
        # docstring lists can cover it. An "unknown"-kind built-in would be
        # undocumentable under either header.
        unknown = {n for n in P.PREDICATE_REGISTRY if P.predicate_kind(n) == "unknown"}
        assert not unknown, f"built-in predicates with unrecognized kind: {sorted(unknown)}"


class TestDocstringMatchesRegistry:
    """The two lists equal the registry grouped by predicate kind."""

    def test_bool_predicate_list_matches_registry(self):
        documented = _doc_names("Available predicates (bool):")
        registered = _registry_names_of_kind("bool")
        assert documented == registered, (
            f"bool docstring drift: missing={sorted(registered - documented)} stale={sorted(documented - registered)}"
        )

    def test_reward_term_list_matches_registry(self):
        documented = _doc_names("Available reward terms (float):")
        registered = _registry_names_of_kind("float")
        assert documented == registered, (
            f"reward-term docstring drift: missing={sorted(registered - documented)} "
            f"stale={sorted(documented - registered)}"
        )
