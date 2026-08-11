# GL-gate non-vacuity pin: keyed on modules, not on an assertion count

- `scenarios.py` - applies the two scenarios (a second gated assertion in a listed
  module; a new module entering scope) to the real tree and reports the guard's verdict.
- `mutations.py` - the two-arm mutation table: every vacuity the two old pins caught
  against the single new pin, plus mutations of the new pin itself.
- `capture.py` / `compose.py` - the frame the four gated assertions verify, rendered
  headless, and the figure. Every rendered number is asserted against `facts.json`.
