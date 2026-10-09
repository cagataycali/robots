### Tests: three backend domain sweeps walk their value tables in one cell

The step-count, pose-vector and replay-episode-index suites gave every value of
a guard's probe table its own cell on every surface that calls the guard. Each
surface now walks the table in one cell and names the failing value in the
assertion. The three files go from 719 cells to 139 with the same package lines
executed, and every planted removal of a call-site guard still fails.
