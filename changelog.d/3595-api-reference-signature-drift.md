### Fixed: the API reference names parameters the callables accept

Three rows of `docs/api-reference.md` documented a parameter that does not
exist, so the first call a reader makes from the table raises:

- `list_robots(category='all')` and `list_robots(category)` - the parameter is
  `mode`, and it filters on backend support (`"all"` / `"sim"` / `"real"` /
  `"both"`), not on category. `list_robots(category='arm')` is a `TypeError`
  and the positional `list_robots('arm')` a `ValueError`; grouping by category
  is `list_robots_by_category()`.
- `register_robot(name, entry)` - there is no `entry`; every field after `name`
  is keyword-only, so even the positional two-argument call is a `TypeError`.

`tests/test_docs_api_reference_signatures_resolve.py` now grades every
signature-shaped code span in the reference against `inspect.signature`, so the
next rename is caught in the file that teaches it rather than at a caller.
