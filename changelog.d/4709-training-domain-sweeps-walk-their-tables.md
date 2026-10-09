### Tests: six training domain sweeps walk their value tables in one cell

The run-size, replay, gradient-clip, TD3-noise and discount-factor suites gave
every value of a domain's probe table its own cell on every backend and field.
Each test now walks the table in one cell and names the failing value in the
assertion, and three "it is reported" tests are folded into the sibling that
already asserts the exact single message. The six files go from 1,457 cells to
242 with the same package lines executed.
