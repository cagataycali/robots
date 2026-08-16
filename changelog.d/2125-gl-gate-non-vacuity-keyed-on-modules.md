### Quality: key the GL-gate guard's non-vacuity pin on modules, not on an assertion count

``tests/test_mujoco_render_assertions_are_gl_gated.py`` held two non-vacuity pins over
one ``frozenset`` of module paths, and the second compared a count of render assertions
to a count of modules. Those are different quantities, equal today only because each of
the four in-scope modules happens to carry exactly one render-success assertion, and
nothing stated that invariant.

The guard's own remedy text tells a contributor to "split the render assertion into its
own case". Doing that inside a module already in scope moved the assertion count and not
the module set, so the required check went red on a diff that complied with the
instruction, reporting ``assert 5 == 4`` - two bare integers naming neither the module
nor the new assertion, and no pin to update.

Both pins are now one, keyed on modules in both directions and reporting the module by
name: an in-scope module that stops contributing a gated assertion, and a gated module
that is not listed. Every vacuity the two old pins caught - a survey that finds nothing,
a scan rooted somewhere unexpected, an assertion that stops being gated - is still
caught. Adding a second correctly gated assertion to a listed module now needs no pin
updated; a new module entering scope still fails, and now names itself.
