### Tests: the nested pytest over the device-connect files runs without the parent's coverage hook

`pytest-cov` hands its measurement to every child interpreter through
`COV_CORE_*` and a `.pth` file that starts coverage at interpreter start,
before the child's own `pytest` reads `--no-cov`. The two cells of
`tests/test_device_connect_stand_in_is_not_handed_back.py` that re-run real
test files in a child paid the parent's whole measurement a second time, for
lines the parent's own session already covers: 9.6 s -> 44.2 s locally with
only the three variables added, and 124 s of a 2642 s CI suite, the largest
single file in the run (#3869). The nested run's environment now drops the
`COV_CORE_` names, pinned on the nested run itself with a control showing the
hook is real. The same file measures 21 s where it measured 86 s.
