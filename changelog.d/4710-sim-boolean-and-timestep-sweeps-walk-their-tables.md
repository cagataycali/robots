### Tests: the boolean-validator and timestep sweeps walk their value tables in one cell

The simulation boolean-validator suite and the timestep-domain suite gave every
value of a probe table (each boolean spelling, each usable and unusable dt) its
own cell on every surface. Each test now walks the table in one cell and names
the failing value in the assertion. The two files go from 420 cells to 107 with
the same package lines executed.
