# no-GL-context contract on `get_frame`

Artifacts for the PR that pins the no-OpenGL-context contract on `get_frame`,
the one consumer of `RenderingMixin._get_renderer` whose channel is a raise.

* `no_gl_context_contract.png` -- the figure embedded in the PR.
* `capture.py` -- drives every consumer with `_get_renderer` forced to `None`,
  renders one real `get_frame` frame on a GL-capable host, and measures the
  envelope-unpack hazard. Writes `facts.json` + `frame.npy`.
* `compose.py` -- builds the figure. Re-derives every rendered number from
  `facts.json` / `mutations.json` and asserts it, plus layout guards.
* `mutation_table.py` -- the 6 plausible regressions, each AST-scoped to its
  enclosing function, run against both arms (the new tests, and the 290 tests
  already covering `get_frame` / the renderer-None family / the compositor).
  Restores both sources in a `finally` and asserts byte-identity.
* `mutations.json` -- the measured mutation results.

Reproduce: `MUJOCO_GL=egl python3 capture.py && python3 compose.py` from a
checkout with the PR applied.
