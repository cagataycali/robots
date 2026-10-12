### Tests: training domain suites drop the cells that ran no package code

67 cells across 19 `tests/training/test_<field>_domain.py` suites asserted what
`torch`, `peft`, `numpy`, `json` or plain Python arithmetic does with a value the
gate already refuses (`len(range(True)) == 1`, `clip_grad_norm_` with `nan`).
Coverage contexts show each of them executed zero lines of `strands_robots`, so
they go. Cells that read the package's own source, or drive package code, stay.
`test_seed_domain.py` and `test_td3_noise_domain.py` had nothing else and are
removed; the seed rationale moves onto `seed_problems`. 1,782 cells become 1,715.
