### Added: docs numbers hook - counts the docs never type by hand

`docs/hooks/numbers.py` substitutes `{{n:robots}}`, `{{n:categories}}`,
`{{n:aliases}}`, `{{n:arm}}` ... at `mkdocs build` from `robots.json` and the
package tree; an unknown key fails `--strict`. The pages that quoted a robot
count by hand (68 on one, 73 on another) now derive it from the registry.
`tests/test_docs_numbers_hook_keys_resolve.py` grades the tokens; the two
catalog-coverage tests that pinned the typed literals are removed (they graded
prose the hook now generates - test bloat, harness #429).
