"""
Repro: docs/learn/policies/microduck.md:71 tells users to `find scene_rollers.xml
under microduck/ on get_search_paths()` but never says WHICH module exports it.

A user copying the surrounding sketches (which do show `from strands_robots.policies ...`)
will guess the two most-nearby-looking namespaces first — and both fail:

    from strands_robots import get_search_paths          -> ImportError
    from strands_robots.registry import get_search_paths -> ImportError
    from strands_robots.policies import get_search_paths -> ImportError

Actual location: `strands_robots.utils.get_search_paths` (also re-exported by `strands_robots.assets`).
The microduck policy page is the ONLY docs mention of `get_search_paths()` — every other public
import on that page is shown with its `from ... import` line. This one is bare.

Fix: either add `from strands_robots.assets import get_search_paths` to line 71,
or re-export it from `strands_robots.registry` (nearest neighbour to the surrounding sketches).
"""

# Reproduce the three guesses a new user would try, in order:
attempts = [
    "from strands_robots import get_search_paths",
    "from strands_robots.registry import get_search_paths",
    "from strands_robots.policies import get_search_paths",
]

for stmt in attempts:
    try:
        exec(stmt)
        print(f"OK:   {stmt}")
    except ImportError as e:
        print(f"FAIL: {stmt}   -> ImportError: {e}")

# And the actually-working ones:
print()
print("Working:")
from strands_robots.utils import get_search_paths as gsp_utils
from strands_robots.assets import get_search_paths as gsp_assets
print("  from strands_robots.utils   import get_search_paths  ->", gsp_utils)
print("  from strands_robots.assets  import get_search_paths  ->", gsp_assets)
