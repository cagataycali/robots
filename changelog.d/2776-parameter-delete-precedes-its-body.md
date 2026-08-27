### Tests: a parameter `del` that narrows scope for nothing is refused locally

`del <unused parameters>` is the tree's idiom for a signature a caller supplies
and the implementation ignores - an agent-tool `stream`, a `HardwareDriver` verb
a platform does not implement, a `Policy.reset` that takes no seed. Twenty sites
use it across `strands_robots/`, `tests/` and `examples/`, and it carries a
property nothing was checking: the `del` has to come *before* the statements it
narrows scope for. As a function's last statement it narrows nothing, because
the frame is discarded on return either way, so it is a no-op statement rather
than a scope-narrowing one.

That shape is not a style preference here. CodeQL reports it as
`py/unnecessary-delete`, and `.github/codeql/codeql-config.yml` records why an
alert is a hard gate - `github-advanced-security` opens a review thread per new
alert and the `default` ruleset sets `required_review_thread_resolution: true`,
so severity never enters into it. That same file states the no-op-statement
capability given up by excluding `py/ineffectual-statement` "is NOT given up --
it moves to ruff, which is merge-blocking here where CodeQL is advisory". For
this shape only the first half holds: measured against ruff 0.15, no ruff rule
reports a terminal parameter `del`, under the repo's `select` list or under
`--select ALL`, because `B015` covers a useless comparison and `B018` a useless
expression and a `del` is neither statement form. So the class reaches the merge
gate having passed `ruff`, `mypy` and `pytest`, and arrives as a review thread
after a push rather than as a local failure.

`tests/test_parameter_deletes_precede_the_body_they_narrow.py` is the missing
local half. It walks every top-level directory that ships Python, deriving that
list rather than restating it so a directory added later is graded on arrival,
and reports each offender as a path, line, function and the parameter names, so
the failure names the edit instead of the rule. The tree satisfies the rule at
twenty of twenty sites, which means the scan finds nothing and cannot exercise
its own failing branch; two classes stand in for that. A non-vacuity class
asserts the scan really sees those twenty across several files and areas, so the
rule cannot pass by having stopped matching the idiom. A constructed-shape class
drives the same predicate over ten exemplars written in the file, including the
constructs it deliberately leaves alone: a `del` of a local, which is the
load-bearing refcount probe used under `tests/simulation/` to make an object
collectable, a `del` of an attribute or a subscript, which are real deletions, a
`del` mixing a parameter with either of those, and a `del` nested inside a
branch, which is a different construct from the docstring-then-`del` shape the
idiom takes.
