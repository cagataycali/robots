### Tests: six more training domain sweeps walk their value tables in one cell

The temperature learning-rate, RL env-count, posture-flag, initial-temperature,
learning-rate and RL checkpoint-interval suites gave every value of a domain's
probe table its own cell. Each test now walks the table in one cell and names
the failing value in the assertion. The six files go from 784 cells to 192 with
the same package lines executed.
