"""Guard against phantom policy providers advertised to agents and users.

The MuJoCo ``run_policy`` tool spec (``tool_spec.json``) and the
``strands_robots.registry.policies`` docstrings enumerate example policy
provider names to guide LLM agents and library callers. If any advertised
name is not actually registered, the caller gets an ``Unknown policy
provider`` error late in a rollout (after world init / camera setup) or is
silently redirected to the ``lerobot_local`` fallback -- a wasteful, confusing
failure mode.

These tests pin the invariant that every provider name advertised in the tool
spec and in the resolver docstrings resolves to a registered provider, so
documentation cannot drift ahead of the implementation again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from strands_robots.registry.policies import list_policy_providers, resolve_policy

ROOT = Path(__file__).resolve().parents[2]
TOOL_SPEC = ROOT / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"


def _registered() -> set[str]:
    return set(list_policy_providers())


def _eg_names(text: str) -> list[str]:
    """Extract provider tokens from an ``e.g. a, b, c`` clause in ``text``."""
    match = re.search(r"e\.g\.\s*([^).]*)", text)
    if not match:
        return []
    return [tok.strip() for tok in re.split(r"[,/]", match.group(1)) if tok.strip()]


def _tool_spec_provider_examples() -> list[str]:
    spec = json.loads(TOOL_SPEC.read_text())
    desc = spec["properties"]["policy_provider"]["description"]
    return _eg_names(desc)


def test_tool_spec_advertises_provider_examples() -> None:
    """The tool spec description must list concrete example providers."""
    assert _tool_spec_provider_examples(), "no 'e.g. ...' provider examples found in tool_spec description"


def test_tool_spec_provider_examples_are_registered() -> None:
    """Every provider named in the MuJoCo tool spec must be registered."""
    registered = _registered()
    unknown = [name for name in _tool_spec_provider_examples() if name not in registered]
    assert not unknown, f"tool_spec advertises unregistered providers {unknown}; registered={sorted(registered)}"


def test_tool_spec_provider_examples_resolve_to_themselves() -> None:
    """Advertised names must resolve to their own provider, not the fallback.

    ``resolve_policy`` redirects any unrecognised string to ``lerobot_local``.
    A phantom name (e.g. ``lerobot_async``) therefore "resolves" but to the
    wrong provider, silently misdirecting the caller. Asserting the resolved
    provider equals the advertised name catches that.
    """
    for name in _tool_spec_provider_examples():
        provider, _ = resolve_policy(name)
        assert provider == name, f"advertised provider {name!r} resolves to {provider!r} (phantom/misdirecting)"


def test_resolve_policy_docstring_examples_reference_registered_providers() -> None:
    """The ``# -> ("name", ...)`` examples in resolve_policy's docstring must be real."""
    doc = resolve_policy.__doc__ or ""
    names = re.findall(r'#\s*\u2192\s*\("([^"]+)"', doc)
    assert names, "no docstring resolution examples found in resolve_policy"
    registered = _registered()
    unknown = [name for name in names if name not in registered]
    assert not unknown, f"resolve_policy docstring references unregistered providers {unknown}"
