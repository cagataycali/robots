"""Repo hygiene: the policy overview table lists exactly the registered providers.

``docs/policies/overview.md`` is the entry point for choosing a
``policy_provider``. Its "Providers" table is the human-facing catalogue of what
``create_policy("<name>")`` accepts. If a provider is added to
``strands_robots/registry/policies.json`` but not to the table (or vice versa),
the docs silently drift and users - and agent LLMs reading the docs - never
discover the provider. This guard ties the table to the JSON registry so the
two can never disagree.

The registry is read directly from ``policies.json`` (not via
``list_providers()``) because ``list_providers()`` also returns providers
registered at runtime via ``register_policy()``. Other tests in the suite
register throwaway providers into that process-global runtime registry and do
not always tear them down, so ``list_providers()`` is order-dependent under a
full ``pytest`` run. The JSON file is the canonical, stable catalogue the
overview table documents. This mirrors ``_registered_providers()`` in the
companion guard.

Companion guard: ``tests/test_docs_policy_nav_coverage.py`` ties the registry to
the mkdocs nav (one page per provider). This file ties it to the overview table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERVIEW_MD = REPO_ROOT / "docs" / "policies" / "overview.md"
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"


def _registered_providers() -> set[str]:
    """Return the canonical provider names from the JSON registry.

    Reads ``policies.json`` directly so the set is stable regardless of what
    other tests register into the runtime registry during a full suite run.
    """
    data = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))
    return set(data["providers"].keys())


def _table_providers() -> set[str]:
    """Extract provider names from the Providers table in overview.md.

    Scoped to the ``## Providers`` section so unrelated tables elsewhere on the
    page cannot leak in. Provider names are the backtick-wrapped token in the
    first column (they may be wrapped in a markdown link, e.g. ``[`mock`](...)``).
    """
    text = OVERVIEW_MD.read_text(encoding="utf-8")

    # Isolate the "## Providers" section (up to the next top-level "## ").
    match = re.search(r"\n## Providers\b(.*?)(?:\n## |\Z)", text, re.DOTALL)
    assert match, "overview.md is missing a '## Providers' section"
    section = match.group(1)

    providers: set[str] = set()
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        first = cells[0]
        # Skip the header and the |---| separator rows.
        if first in {"Provider", ""} or set(first) <= {"-", ":"}:
            continue
        name = re.search(r"`([a-z0-9_]+)`", first)
        if name:
            providers.add(name.group(1))
    return providers


def test_overview_table_matches_registered_providers() -> None:
    """The Providers table lists exactly the providers in the JSON registry."""
    documented = _table_providers()
    registered = _registered_providers()

    missing_from_docs = sorted(registered - documented)
    stale_in_docs = sorted(documented - registered)

    assert not missing_from_docs and not stale_in_docs, (
        "docs/policies/overview.md Providers table is out of sync with "
        "strands_robots/registry/policies.json.\n"
        f"  Registered but missing from the table: {missing_from_docs}\n"
        f"  In the table but not registered:       {stale_in_docs}\n"
        "Update the table (name, class, install extra, when-to-use) so it "
        "matches the registry."
    )


def test_overview_has_discovery_snippet() -> None:
    """Overview keeps a runnable ``list_providers()`` snippet for ground-truthing."""
    text = OVERVIEW_MD.read_text(encoding="utf-8")
    assert "list_providers" in text, (
        "overview.md must show a runnable list_providers() snippet so users can "
        "always ground-truth the available providers against the registry."
    )
