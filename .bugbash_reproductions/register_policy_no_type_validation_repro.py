#!/usr/bin/env python3
"""
Repro: `strands_robots.policies.register_policy` performs zero input
validation. The sibling `strands_robots.registry.user_registry.register_robot`
TypeErrors / ValueErrors a bad shape before writing it. This script shows four
distinct families of corruption `register_policy` accepts silently, two of
which turn `create_policy` into a silent-wrong seam.

Source of truth (strands_robots/policies/factory.py:43-63 on HEAD 31a68948):

    def register_policy(name, loader, aliases=None):
        _runtime_registry[name] = loader
        if aliases:
            for alias in aliases:
                _runtime_aliases[alias] = name

No type checks. No callable check on `loader`. No Policy-subclass check on the
`loader()` return. No hashable check on `name` or members of `aliases`.

Sibling (strands_robots/registry/user_registry.py:270-293):

    if hardware is not None and not isinstance(hardware, dict):
        raise TypeError(...)
    if model_xml is not None and not isinstance(model_xml, str):
        raise TypeError(...)
    name = normalize_robot_name(name)  # rejects non-str

The two surfaces document the same promise (`register_X` fills a user
registry) with asymmetric guard strength.

Run:
    python3 register_policy_no_type_validation_repro.py
"""
from __future__ import annotations

from strands_robots.policies.base import Policy
from strands_robots.policies.factory import (
    create_policy,
    register_policy,
    _runtime_aliases,
    _runtime_registry,
)


def rule(title: str) -> None:
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


# --- Family A: loader() returns a NON-Policy class ------------------------
# User-plausible mistake: forgot `class MyPolicy(Policy):` — wrote `class MyPolicy:`.
rule("A. create_policy returns a non-Policy — signature `-> Policy` violated")


class NotAPolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


register_policy("buggy_a", lambda: NotAPolicy)
p = create_policy("buggy_a", foo="bar")
print(f"  create_policy('buggy_a') -> {type(p).__name__}")
print(f"  isinstance(p, Policy) == {isinstance(p, Policy)}  <-- signature promise broken")
# Downstream: SimEngine.run_policy(..., policy_object=p) later calls
# p.get_actions / p.reset / p.name and crashes with AttributeError that
# names factory internals, not the actual bug (forgot `(Policy)`).
for attr in ("get_actions", "reset", "name", "requires_images"):
    print(f"    hasattr(p, {attr!r}) == {hasattr(p, attr)}")

# --- Family B: loader is not callable -------------------------------------
rule("B. loader=None / loader=str accepted; TypeError at build time names internals")

register_policy("buggy_b_none", None)
try:
    create_policy("buggy_b_none")
except TypeError as e:
    print(f"  create_policy -> TypeError: {e}")
    # The error names 'NoneType' — not 'register_policy'. User has to
    # reason backwards from "'NoneType' object is not callable" to "oh,
    # the loader I passed was None, and nobody warned me".

register_policy("buggy_b_str", "not_a_callable")
try:
    create_policy("buggy_b_str")
except TypeError as e:
    print(f"  create_policy -> TypeError: {e}")

# --- Family C: non-string name poisons the runtime registry ----------------
rule("C. non-string name accepted; _runtime_registry keyed by int/None/bool/bytes")

for n in [None, 123, True, b"wbc"]:
    register_policy(n, lambda: NotAPolicy)
    print(f"  register_policy({n!r}, ...) accepted; "
          f"{n!r} in _runtime_registry = {n in _runtime_registry}")

# --- Family D: non-string aliases silently poison _runtime_aliases ---------
rule("D. aliases=[123, None, 'ok'] accepted; _runtime_aliases now has non-str keys")

register_policy("buggy_d", lambda: NotAPolicy, aliases=[123, None, "ok"])
print(f"  123 in _runtime_aliases = {123 in _runtime_aliases}")
print(f"  None in _runtime_aliases = {None in _runtime_aliases}")
print(f"  'ok' in _runtime_aliases = {'ok' in _runtime_aliases}")
# A later list_aliases() returns a dict with mixed-type keys; any JSON
# serialiser (telemetry, docs tooling) or dict-sort hits TypeError.

# --- Compare: register_robot refuses the same shapes ----------------------
rule("E. SIBLING `register_robot` refuses what `register_policy` accepts")

from strands_robots.registry.user_registry import register_robot

for shape, kwargs in [
    ("hardware=42 (non-dict)", dict(name="probe_a", hardware=42)),
    ("model_xml=123 (non-str)", dict(name="probe_b", model_xml=123)),
    ("name=None", dict(name=None, category="arm")),
]:
    try:
        register_robot(**kwargs)
        print(f"  register_robot({shape}): accepted (should refuse)")
    except (TypeError, ValueError, AttributeError) as e:
        print(f"  register_robot({shape}): refused -> {type(e).__name__}: {str(e)[:90]}")

print(
    "\nFix shape (mirrors register_robot, ~20 LOC):\n"
    "  * TypeError if name not str (or hashable only)\n"
    "  * TypeError if loader not callable\n"
    "  * TypeError if any alias not str\n"
    "  * At create_policy() seam: assert issubclass(loader(), Policy) -> ValueError naming the register_policy site\n"
    "No existing test pins the current permissive behaviour (grep register_policy tests/)."
)
