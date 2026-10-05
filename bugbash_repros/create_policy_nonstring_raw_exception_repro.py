"""Repro: create_policy() leaks raw CPython exceptions for non-string ``provider``.

Target: strands-labs/robots v0.5.3

Symptom
-------
``create_policy(provider)`` is declared ``provider: str`` and sibling
``provider_can_be_created(provider: Any) -> bool`` on the same file is a
preflight that answers *the same question* for *the same input domain*
(its own docstring, factory.py:755: "answer exactly the question
:func:`create_policy` answers"). The preflight gracefully returns
``False`` for every non-string type; the dispatcher it fronts leaks three
different internal CPython errors depending on which stage of the walk
touches the value first:

- ``None`` / ``int`` / ``bool`` / ``Policy`` instance / ``Policy`` class
  -> ``AttributeError: '<type>' object has no attribute 'strip'`` from
  ``_is_smart_string`` at factory.py:245.
- ``dict`` / ``list``
  -> ``TypeError: unhashable type: 'dict'`` from the alias lookup at
  factory.py:451 (``_runtime_aliases.get(provider, provider)``).
- ``bytes``
  -> ``TypeError: a bytes-like object is required, not 'str'`` from the
  substring probe at factory.py:246.

None of the three errors names ``provider``, ``create_policy``, or the
expected type; none offers a fix. The sibling validator
``policy_object_error`` on the same file demonstrates the correct message
shape for this exact mistake (passing a Policy-shaped thing to a slot
that wants a string), and the sibling preflight
``provider_can_be_created`` demonstrates the correct domain handling
(``Any`` in, ``bool`` out, no raise).

Fix is at ``strands_robots/policies/factory.py:771`` (``create_policy``
entry): refuse non-string at the top with a typed message naming the
parameter, the type received, and the expected shape. ~10 LOC. The
preflight at :254 already does the equivalent check cheaply, so the
dispatcher cannot promise less than it does.

Not a duplicate of:
- harness#679 (create_policy unknown-string provider did-you-mean)
- harness#685 (URL-scheme swallow)
- harness#690 (trust-remote-code gate ordering)
- harness#697 (punctuation reroute to lerobot_local)
- harness#717 (create_policy("persistent") -- a VALID provider name
  colliding with positional kwarg; distinct code path at the policy
  class, not the factory entry).
"""

from __future__ import annotations

import inspect
import sys
import traceback

from strands_robots.policies import create_policy, policy_object_error
from strands_robots.policies.base import Policy
from strands_robots.policies.factory import provider_can_be_created
from strands_robots.policies.mock import MockPolicy


# ---------------------------------------------------------------------------
# Inputs: a cross-section of the non-string domain the sibling preflight accepts.
# ---------------------------------------------------------------------------
INSTANCE = MockPolicy()
NON_STRING_INPUTS: list[tuple[str, object]] = [
    ("None", None),
    ("int (42)", 42),
    ("bool (True)", True),
    ("bytes (b'mock')", b"mock"),
    ("dict ({'x': 1})", {"x": 1}),
    ("list ([])", []),
    ("Policy class (MockPolicy)", MockPolicy),
    ("Policy instance (MockPolicy())", INSTANCE),
]


def _bottom_frame(exc: BaseException) -> str:
    """Last stack line of ``exc``'s traceback."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    for line in reversed(tb):
        line = line.strip()
        if line.startswith("File "):
            return line
    return tb[-1].strip() if tb else ""


def _call(fn, *args, **kwargs):
    """Return ("ok", value) or ("raised", exc)."""
    try:
        return "ok", fn(*args, **kwargs)
    except BaseException as e:  # noqa: BLE001 - we want everything, including CPython builtins
        return "raised", e


# ---------------------------------------------------------------------------
# Section 1: sibling preflight is domain-correct.
# ---------------------------------------------------------------------------
print("=" * 76)
print("1. provider_can_be_created(Any) -> bool   (factory.py:254)")
print("=" * 76)
preflight_sig = inspect.signature(provider_can_be_created)
print(f"signature: {preflight_sig}")
preflight_outcomes = {}
for label, val in NON_STRING_INPUTS:
    outcome, result = _call(provider_can_be_created, val)
    preflight_outcomes[label] = (outcome, result)
    tag = "RAISED" if outcome == "raised" else f"returned {result!r}"
    print(f"  {label:40s} -> {tag}")

print()
assert all(o == "ok" and r is False for o, r in preflight_outcomes.values()), (
    "preflight should return False cleanly for every non-string input"
)
print("  OK: preflight returns False for every non-string input, raises nothing.")

# ---------------------------------------------------------------------------
# Section 2: create_policy leaks raw CPython exceptions for the same inputs.
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("2. create_policy(str) for the SAME inputs   (factory.py:771)")
print("=" * 76)
dispatcher_sig = inspect.signature(create_policy)
print(f"signature: {dispatcher_sig}")

create_outcomes = []
for label, val in NON_STRING_INPUTS:
    outcome, result = _call(create_policy, val)
    assert outcome == "raised", f"{label}: expected a raise, got {result!r}"
    create_outcomes.append((label, result))
    print(f"\n  {label}")
    print(f"    -> {type(result).__name__}: {result}")
    print(f"    {_bottom_frame(result)}")

print()
leaked_types = {type(exc).__name__ for _, exc in create_outcomes}
print(f"  Exception types: {sorted(leaked_types)}")

# Two modes:
#   PRE-FIX  (defect state):  AttributeError + TypeError, no message names
#                             `provider`, `create_policy`, or the expected type.
#   POST-FIX (fixed state):   TypeError only, every message names all three.
all_mention_provider = all("provider" in str(exc).lower() for _, exc in create_outcomes)
all_mention_create_policy = all(
    "create_policy" in str(exc).lower() for _, exc in create_outcomes
)
all_typeerror = leaked_types == {"TypeError"}
if all_typeerror and all_mention_provider and all_mention_create_policy:
    print("  FIX IS IN: every refusal is a TypeError naming `provider` and `create_policy`.")
    FIX_IS_IN = True
else:
    assert leaked_types == {"AttributeError", "TypeError"}, (
        "pre-fix spec: raw CPython AttributeError/TypeError only"
    )
    for label, exc in create_outcomes:
        msg = str(exc).lower()
        assert "provider" not in msg, f"{label}: unexpectedly mentions 'provider'"
        assert "create_policy" not in msg, f"{label}: unexpectedly mentions 'create_policy'"
        assert "expected" not in msg, f"{label}: unexpectedly mentions 'expected'"
    print("  DEFECT STATE: no message names `provider`, `create_policy`, or expected type.")
    FIX_IS_IN = False

# ---------------------------------------------------------------------------
# Section 3: sibling validator on the same file shows the correct shape.
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("3. policy_object_error(Any) -> str | None   (factory.py:516, same file)")
print("=" * 76)
print("Same inputs, typed message naming the parameter and the fix:")
for label, val in NON_STRING_INPUTS:
    msg = policy_object_error(val)
    one_line = (msg[:140] + "...") if msg and len(msg) > 140 else msg
    print(f"  {label:40s} -> {one_line}")

# ---------------------------------------------------------------------------
# Section 4: documentation on the preflight explicitly promises symmetry.
# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("4. Docstring promise (factory.py:254-275)")
print("=" * 76)
doc = inspect.getdoc(provider_can_be_created) or ""
excerpt = next(
    (ln for ln in doc.splitlines() if "same" in ln.lower() and "question" in ln.lower()),
    "",
)
print(f"  '{excerpt.strip()}'")
assert excerpt, (
    "docstring explicitly claims the preflight answers the SAME question "
    "create_policy answers; today it answers a strictly larger domain"
)
print("  Preflight domain: Any (returns False, no raise).")
print("  Dispatcher domain today: str only (raw AttributeError/TypeError on everything else).")
print("  Expected after fix: TypeError naming `provider`, type received, and expected shape --")
print("  mirroring policy_object_error's wording.")

print()
print("DEFECT REPRODUCED." if not FIX_IS_IN else "FIX VERIFIED.")
sys.exit(0)
