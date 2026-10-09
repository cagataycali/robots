### Tests: the rate-agreement parity grid is one cell, not 225

`requested_rate_mismatch_reason`'s parity against the two rate domains was a
15 x 15 parametrized grid plus a second test that walked the same grid to check
it reached all three outcomes. Both are now one cell that walks the grid once,
names the failing pair, and asserts the outcome set. The test file goes from
276 cells to 51 with the same package lines executed.
