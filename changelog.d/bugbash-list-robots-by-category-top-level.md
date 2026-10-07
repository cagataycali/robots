### Fixed: `strands_robots.list_robots_by_category` is top-level reachable

The README's hero `What you get` row 1 promises "150+ robots across 8
categories"; the function that answers that split
(`list_robots_by_category`) sits in `strands_robots.registry.__all__`
next to three siblings (`list_robots`, `list_discoverable`,
`list_urdf_only`), but only the three siblings were promoted to the
top-level `strands_robots` namespace. A user who follows the
`docs/start/*.md` convention `from strands_robots import X` hit
`ImportError` and had to deep-import from `strands_robots.registry`.

Three-line additive change: `list_robots_by_category` joins the
`TYPE_CHECKING` block, `_LAZY_IMPORTS`, and `__all__` next to its
siblings. The function is the same callable as before (identity
check: `strands_robots.list_robots_by_category is strands_robots.registry.list_robots_by_category`).

Pinned by `tests/test_list_robots_by_category_is_top_level.py` (4 cases:
import works, in `__all__`, four-sibling sym­metry, output matches the
README's "8 categories / 150+ robots" claim).
