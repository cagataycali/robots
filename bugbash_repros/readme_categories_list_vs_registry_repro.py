"""Repro: README.md:88 names a category list that does not match the registry.

Symptom
=======
README.md:88 says:
    "150+ robots across 8 categories - arms, bimanual rigs, humanoids,
     quadrupeds, hands, drones - from one registry with asset auto-download"

Three distinct mistakes in one line:
    1. "quadrupeds" is NOT a category. The registry uses "mobile" (which
       covers quadrupeds + wheeled bases). A reader clicking into
       docs/robots/index.md to filter for quadrupeds finds no such button.
    2. The number says "8 categories" but only 6 are named.
    3. The 2 the sentence omits -- "expressive" and "mobile_manip" -- do
       exist in the registry AND have dedicated filter buttons on
       docs/robots/index.md (lines 25, 27, 28).

Expected
========
The six/eight story is internally consistent and matches the filter-button
set in docs/robots/index.md. A user who reads the README and clicks "See all"
sees the same vocabulary.

Actual
======
Mismatch verified below.

Run
===
    cd /path/to/robots && python bugbash_repros/readme_categories_list_vs_registry_repro.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from strands_robots.registry import list_robots

REPO = Path(__file__).resolve().parent.parent


def _readme_claim() -> tuple[int, list[str]]:
    """Parse the README line that enumerates categories.

    Returns (claimed_count, named_categories_lowercased).
    """
    txt = (REPO / "README.md").read_text(encoding="utf-8")
    line = next(
        ln for ln in txt.splitlines() if "robots across" in ln and "categories" in ln
    )
    m_count = re.search(r"(\d+)\s+categor", line)
    assert m_count, f"could not parse claimed count from: {line!r}"
    claimed_count = int(m_count.group(1))
    # The names sit between the first " - " (after "categories**") and
    # " - from one registry". Strip markdown first.
    plain = re.sub(r"\*+", "", line)
    m_seg = re.search(r"categories\s+-\s+(.*?)\s+-\s+from one registry", plain)
    assert m_seg, f"could not parse category list from: {plain!r}"
    seg = m_seg.group(1)
    named = [t.strip().lower().rstrip("s") for t in seg.split(",")]
    # "bimanual rigs" -> "bimanual rig" -> "bimanual"; keep head word.
    named = [n.split()[0] for n in named if n]
    return claimed_count, named


def _registry_categories() -> set[str]:
    cats = Counter(r.get("category", "?") for r in list_robots())
    return set(cats.keys())


def _docs_filter_buttons() -> set[str]:
    """What docs/robots/index.md actually offers in its filter row."""
    idx = (REPO / "docs" / "robots" / "index.md").read_text(encoding="utf-8")
    # <button class="sr-filter-btn" data-family="arm" ...>
    return set(re.findall(r'data-family="([a-z_]+)"', idx)) - {"all"}


def main() -> int:
    claimed_count, named = _readme_claim()
    registry = _registry_categories()
    docs_buttons = _docs_filter_buttons()

    print(f"README L88 claims count: {claimed_count}")
    print(f"README L88 lists names:  {named!r}  (n={len(named)})")
    print(f"Registry categories:     {sorted(registry)!r}  (n={len(registry)})")
    print(f"Docs filter buttons:     {sorted(docs_buttons)!r}  (n={len(docs_buttons)})")
    print()

    # Invariant 1: the sentence names as many categories as it claims.
    if claimed_count != len(named):
        print(
            f"[FAIL] README says '{claimed_count} categories' but enumerates only "
            f"{len(named)}: {named!r}"
        )
    else:
        print(f"[OK] README count ({claimed_count}) matches enumeration length.")

    # Invariant 2: every name in the README refers to a real registry category
    # (bimanual == bimanual, quadruped is NOT a registry category -- it's covered
    # by 'mobile'; drones is NOT a category -- it's 'aerial').
    def _canon(n: str) -> str:
        return {
            "drone": "aerial",
            "quadruped": "mobile",
            "mobile": "mobile",  # README might say "mobile" directly
            "aerial": "aerial",
            "expressive": "expressive",
        }.get(n, n)

    bad_names = [n for n in named if _canon(n) not in registry]
    if bad_names:
        print(f"[FAIL] README names non-registry categories: {bad_names!r}")
    else:
        print("[OK] every README name maps to a real registry category.")

    # Invariant 3: no real registry category with its own docs filter button is
    # left out of the README sentence.
    canon_named = {_canon(n) for n in named}
    missing = sorted((registry & docs_buttons) - canon_named)
    if missing:
        print(
            f"[FAIL] README omits registry categories that have dedicated filter "
            f"buttons in docs/robots/index.md: {missing!r}"
        )
    else:
        print("[OK] README covers all registry categories with filter buttons.")

    # The repro passes if all three hold -- i.e. the fix is in place.
    all_ok = (
        claimed_count == len(named)
        and not bad_names
        and not missing
    )
    print()
    if all_ok:
        print("PASS: README L88 enumeration matches registry + docs filter row.")
        return 0
    print("DEFECT: README L88 enumeration does not match registry nor docs filter row.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
