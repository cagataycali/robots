"""Minimal repro: factory.list_providers() leaks runtime aliases into a canonical-only surface.

Expected (per docstring at strands_robots/policies/factory.py:113):
    list_providers() -> list[str]  -- 'List all available policy provider names (JSON + runtime)'

Expected (per sibling in strands_robots/registry/policies.py:316):
    list_policy_providers() -> 'canonical only'

Expected (per published union pattern in list_aliases docstring, factory.py:130):
    registered = set(list_providers()) | set(list_aliases())
    -- i.e. the two sets are disjoint and the union enumerates all spellings.

Observed:
    After register_policy("my_fake", loader, aliases=["fakey"]),
    'fakey' appears in BOTH list_providers() and list_aliases().

Root cause: strands_robots/policies/factory.py:118 includes _runtime_aliases.keys()
in the list that is meant to carry canonical names only.
"""
from strands_robots.policies import register_policy, list_providers, list_aliases
from strands_robots.policies.base import Policy


class _Dummy(Policy):
    pass


register_policy("my_fake", lambda: _Dummy, aliases=["fakey"])

canonicals = set(list_providers())
aliases    = set(list_aliases().keys())
intersect  = canonicals & aliases

print(f"list_providers() size: {len(canonicals)}")
print(f"list_aliases()  size: {len(aliases)}")
print(f"intersection:        {sorted(intersect)}")

assert "fakey" in aliases, "alias registered"
assert "fakey" not in canonicals, (
    f"BUG: alias 'fakey' leaked into list_providers() canonical surface. "
    f"intersection with list_aliases(): {sorted(intersect)}"
)
