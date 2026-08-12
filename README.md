# joint_limits: a non-finite bound the ordering check cannot see

`capture.py` measures, in one tree, the construction verdict for every
`(min, max)` kind plus the command consequence; `compose.py` builds the figure
from the two dumps and asserts every rendered number against them.

Reproduce:

    python3 capture.py out.json     # run once per tree
    python3 compose.py              # composes from base.json + branch.json

`base.json` is `strands-labs/robots@32bab339`, `branch.json` is that commit plus
this change. 16 of 25 construction verdicts were wrong; 0 after.
