### Quality: a comparison written between two literals is refused

`assert True > 0` reads as a measurement and is not one. Both operands are
literals, so the result is fixed when the line is typed: the assertion cannot
fail, cannot notice a change in what it claims to measure, and cannot tell a
correct premise from a mis-transcribed one. Two premise tests carried the shape --
the places whose whole value is that a claim was measured rather than restated --
and both now put the value under test through a name so the comparison is decided
when it runs.

Neither merge gate covered it. `ruff` selects `B015` for this capability, but
`B015` fires only when a comparison's result is unused, so it is silent by design
inside an `assert`, which consumes it; CodeQL's `py/comparison-of-constants`
reported one of the two instances on a pull-request ref while its twin sat on
`main` unreported, through a default-branch analysis that was current and
successful. The shape is now refused deterministically by a repository-wide scan
over every Python file the project ships.
