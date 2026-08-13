# Isaac `run_multi_policy` shared caller-knob refusals

Measurement for the tests-only PR that drives `IsaacSimulation.run_multi_policy`'s four
shared caller-knob refusals (`control_frequency`, `duration`, `instructions`,
`action_horizon`) through the Isaac entry point.

* `capture.py` - re-derives every number in the figure: what each knob reports, whether
  its refusal line was executed before/after, and the coverage delta. Self-audits with
  `assert` on every claim before writing `facts.json`.
* `mutate.py` - the 8-row mutation table, each anchor scoped to `run_multi_policy`'s own
  AST line range (`in_fn` vs `in_file` printed as the justification), source restored
  byte-identically.
* `compose.py` - draws the figure and asserts every drawn number against `facts.json`.
* `facts.json` - the measured payload.

Reproduce:

    MUJOCO_GL=egl PYTHONPATH=. python3 mutate.py     # from the repo root, as _probe/mutate.py
    MUJOCO_GL=egl PYTHONPATH=. python3 capture.py    # needs /tmp/cov-{before,after}-$RUN.json
    MUJOCO_GL=egl python3 compose.py

No Isaac Sim Kit runtime, no GPU and no MuJoCo are needed for any of the new test cases.
