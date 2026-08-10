# eval-facade pre-flight refusals

`preflight_refusals.png` — three headless MuJoCo renders of one `so100` scene:

1. the reference pose,
2. after a real `evaluate_benchmark` run (`status=success`, 150/150 steps,
   11.69% of pixels differ from the reference),
3. after all four refused pre-flights (`video` unknown key, `video fps=0`,
   non-mapping `policy_config`, non-mapping `policy_kwargs`) — 0 of 516,800
   pixels changed and every joint identical to the reference.

The capture script asserted each of those numbers against the measured dump
(`measured_facts.json`) before the figure was written, and the composer
re-asserted them plus a clean 8px border.

`mutate_delete_guard.py` — deletes each of the six guard/`return err` pairs in
turn (AST-scoped to the enclosing function, source restored byte-identically)
and reports how many tests fail in the new module vs. the existing suite.

`mutate_discard_refusal.py` — keeps the rate guard's *call* and discards its
refusal, so the AST parity test that covers it still passes while the guard
does nothing.
