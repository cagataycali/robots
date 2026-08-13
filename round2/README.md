# Round 2 measurements -- PR #2226

Scripts behind the round-2 numbers. Run from a checkout of the PR branch with
`PYTHONPATH=<repo root>` so they measure that tree (each prints the tree it
resolved).

* `mirror.py` -- a mirror of CodeQL `py/empty-except` (single-`pass` handler with
  no explanatory comment in or adjacent to it), validated against alert 899's own
  reported location, which it reproduces at exactly `139:9-28`. Reports the
  finding on the tree as first pushed and nothing on either later tree.
* `measure_redundancy.py` -- asks whether the swallowed `ImportError` was
  load-bearing: mutates the funnel to substitute `MockPolicy` and counts how many
  sibling tests already fail. Answer: 8, because five siblings assert the raise
  with `pytest.raises`.
* `measure_success_mutation.py` -- the same question asked of the funnel's success
  path, to check the data assertion is not the only thing ruling the swap out.
* `mutations.py` -- the two-arm table: each regression applied to
  `import_policy_class` and run against the module before and after the round.
  Anchors are AST-scoped to the enclosing function and the production file is
  restored byte-identically in a `finally`.

Measured outcome: the construct is gone in both later shapes, and the return-path
case widens from one provider to four -- 9 failures under the `MockPolicy`
mutation before, 12 after, with the unmutated control clean on all three.
