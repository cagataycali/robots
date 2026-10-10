### Tests: seven more training domain sweeps walk their value tables in one cell

The learning-starts, validation-episodes, policy-delay, RL posture-flag,
optimization-epochs, launch-topology and checkpoint-cadence suites gave every
value of a domain's probe table its own cell. Each test now walks the table in
one cell and names the failing value in the assertion. The seven files go from
791 cells to 195 with the same package lines executed.
