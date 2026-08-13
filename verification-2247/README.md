# Independent verification of strands-labs/robots#2247 on Thor

`capture.py` measures the defect and both verdicts; the `main` arm is read from
`git show upstream/main:scripts/check_merge_base_overlap.py` and the `branch`
arm from the tree it runs in, so the figure describes whatever head it is run
against. `mutations.py` re-applies #2247's three mutation rows, AST-scoped to
the enclosing function, and reports the failing test names. `compose.py` draws
the figure and asserts every rendered number against `facts.json`.

Measured against #2247's head 939cd2382c3b on base 7954c914d3c8.
