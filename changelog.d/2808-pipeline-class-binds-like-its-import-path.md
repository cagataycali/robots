### Fixed: a `cosmos_transfer` pipeline class now binds the same way its import path binds it

The video2video pipeline seam accepts a backend two ways - as an object, or as
a dotted import path resolved lazily - and only one of them applied the
construction step. The dotted-path branch constructs a class or zero-arg
factory target before probing it, because an unconstructed class passes the
`generate()` probe as a plain function and then receives the video as its
`self`. The object branch skipped that step, so the same class bound cleanly
through one form and failed through the other: `validate()` returned no
problems for `pipeline=Adapter`, and generation then raised
`TypeError: Adapter.generate() missing 1 required positional argument: 'video'`
from inside the run - neither the `error` envelope `transform()` returns for a
wiring problem nor the `ValueError` it documents, and after the source dataset
was opened and an output recorder created.

`validate()` exists so a wiring mistake is reported before any of that work
happens, and the documented adapter recipe is spelled `class Adapter:`, so
`Adapter` against `Adapter()` is a one-character slip. Construction is now a
single owner both branches consult, so neither can re-derive it. A class whose
`generate` is a `staticmethod` or a `classmethod` already bound usably and
still does; a class whose constructor needs arguments now names its
constructor rather than a missing `video`.
