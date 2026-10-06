"""Repro: strands_robots.registry read API leaks raw CPython exceptions on non-str inputs.

Nine sibling functions exported from `strands_robots.registry` (which the
package docstring in `strands_robots/registry/__init__.py` lists as "the
public read API") share a type-coercion footgun: each funnels its `name`
argument through `normalize_robot_name(name)`, whose body is
`name.lower().strip().replace("-", "_")` with no type validation
(strands_robots/registry/loader.py:66, L89).

`Robot(name)` already patched this symptom at the door with a `ValueError`
(strands_robots/robot.py:798-807) and a corresponding regression test
(tests/test_a_robot_name_that_is_not_a_string_is_refused_by_name.py). The
in-code comment at robot.py:800-803 explicitly admits the registry layer
leaks the same `AttributeError`/`TypeError`:

    # Refused here, at the door, with the wording the empty string gets: the
    # name reaches ``normalize_robot_name``'s ``.lower()`` before any other
    # check, so ``Robot(None)`` used to escape as an ``AttributeError`` and
    # ``Robot(b"so101")`` as a ``TypeError`` from a regex, neither of which
    # names what to pass instead.

But every caller who imports from `strands_robots.registry` directly (the
documented public read surface) still gets that leak. Expected: all 9
siblings return `None`/`False`/`""`/`{}` or raise a structured `ValueError`
matching `Robot()`'s wording. Sibling asymmetry across the SAME package:

| function                               | file:line                               | non-str outcome        |
|----------------------------------------|-----------------------------------------|------------------------|
| get_robot                              | registry/robots.py:97                   | AttributeError leak    |
| resolve_name                           | registry/robots.py:60                   | AttributeError leak    |
| has_sim                                | registry/robots.py:138                  | AttributeError leak    |
| has_hardware                           | registry/robots.py:160                  | AttributeError leak    |
| joint_labels                           | registry/robots.py:122                  | AttributeError leak    |
| get_driver                             | registry/robots.py:197                  | AttributeError leak    |
| get_hardware_type                      | registry/robots.py:181                  | AttributeError leak    |
| is_discoverable                        | registry/discovery.py                   | AttributeError leak    |
| is_urdf_only                           | registry/discovery.py                   | AttributeError leak    |
| **lerobot_from_source_entry**          | registry/robots.py:214                  | OK (returns None)      |
| **get_policy_provider**                | registry/policies.py:56                 | OK (returns None)      |
| **resolve_policy**                     | registry/policies.py:430                | OK (returns None)      |

The three siblings that work either skip `normalize_robot_name` or route
through a dict lookup (`_canonical_provider_name`).

Run:
    cd <repo>
    python bugbash_registry_api_nonstring_repro.py
"""

from __future__ import annotations

from strands_robots.registry import (
    get_driver,
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    joint_labels,
    resolve_name,
)
from strands_robots.registry.discovery import is_discoverable, is_urdf_only
from strands_robots.registry.robots import lerobot_from_source_entry

LEAKY = [
    ("get_robot", get_robot),
    ("resolve_name", resolve_name),
    ("has_sim", has_sim),
    ("has_hardware", has_hardware),
    ("joint_labels", joint_labels),
    ("get_driver", get_driver),
    ("get_hardware_type", get_hardware_type),
    ("is_discoverable", is_discoverable),
    ("is_urdf_only", is_urdf_only),
]
SIBLING_OK = [("lerobot_from_source_entry", lerobot_from_source_entry)]

BAD_INPUTS = [None, 42, 3.14, True, b"so100", [], {}]

print("=== LEAKY (public read API in strands_robots.registry) ===")
leaked = 0
for name, fn in LEAKY:
    for arg in BAD_INPUTS:
        try:
            fn(arg)
            outcome = "OK (unexpected)"
        except (AttributeError, TypeError) as e:
            outcome = f"{type(e).__name__}: {e}"
            leaked += 1
        except Exception as e:
            outcome = f"{type(e).__name__}: {e}"
        print(f"  {name}({arg!r:<10}) -> {outcome}")

print(f"\n=== SIBLING THAT HANDLES NON-STR GRACEFULLY ===")
for name, fn in SIBLING_OK:
    for arg in BAD_INPUTS:
        r = fn(arg)
        print(f"  {name}({arg!r:<10}) -> {r!r}")

print(f"\nLeaky calls: {leaked} / {len(LEAKY) * len(BAD_INPUTS)}")
assert leaked == len(LEAKY) * len(BAD_INPUTS), "all leaky calls should raise"
print("DEFECT REPRODUCED.")
