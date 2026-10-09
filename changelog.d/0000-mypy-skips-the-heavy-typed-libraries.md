### Tests: the lint step's mypy no longer analyses transformers, diffusers, warp and google

These four ship `py.typed`, so `mypy strands_robots tests tests_integ` read
their whole source on every run, for a handful of imports the package makes
inside function bodies. A `follow_imports = "skip"` override reads them as
`Any` instead: a cold run on one core takes 125 s instead of 190 s and peaks
at 1.9 GB instead of 2.9 GB, with the same findings. A new check keeps every
import of a skipped library inside a function, so none of them can put an
unchecked type in a signature.
