### Tests: six more training domain sweeps walk their value tables in one cell

The LoRA, network-width, Polyak, clip-range, GAE-lambda and loss-weight suites
gave every value of a domain's probe table its own cell on every backend and
field. Each test now walks the table in one cell and names the failing value in
the assertion. The six files go from 896 cells to 181 with the same package
lines executed.
