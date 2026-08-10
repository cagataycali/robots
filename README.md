# lerobot install-probe contract - measurement artifacts

`capture.py` measures every fact the figure states and writes `measured_facts.json`;
`compose.py` reads only that JSON and asserts each rendered number before saving
`probe_contract.png`. Run `capture.py` from a checkout of the branch.

`capture.py` mutates `strands_robots/dataset_recorder.py` in place for the mutation
matrix and restores it byte-identically in a `finally`.
