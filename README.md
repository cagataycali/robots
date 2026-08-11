# Recording lifecycle guard parity — measurement artifact

`capture.py` derives every number in the figure: the cross-backend contract matrix
(each contract located by anchor text, then looked up in the coverage JSON), the
per-file coverage delta, and the 6x2 mutation table (re-run, both arms). `compose.py`
renders it and asserts each rendered value against `facts.json` before saving.

Reproduce from a checkout of the PR branch:

    MUJOCO_GL=egl python3 -m pytest tests -q --cov=strands_robots \
        --cov-report=json:/tmp/cov-after.json
    python3 capture.py && python3 compose.py
