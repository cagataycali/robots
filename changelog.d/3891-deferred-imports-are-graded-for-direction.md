### Tests: a deferred upward import is graded like a module-scope one

`scripts/check_import_layers.py` graded direction on module-scope imports only,
so `upward runtime edges: 0` read as "nothing points up" while 16 inversions sat
inside function bodies. Deferring an import moves *when* a dependency is paid,
not whether it exists, so direction is now graded on the deferred graph too,
against its own ratchet `KNOWN_DEFERRED_UPWARD_EDGES` -- an equality, so a new
hidden inversion fails until it is declared and a removed one fails until its
line goes. Acyclicity still exempts late imports: that property is about
import-time mechanics, and a late import is the sanctioned way to break a cycle.
Typing-only imports are reported and graded by neither.

One inversion goes with it. `hardware.driver` is a registry field and the loader
refuses a value outside `DRIVER_CHOICES` at load time rather than leaving every
reader to re-check it, but the vocabulary lived in the driver seam a layer up, so
the layer that validates a declared entry reached up for the list of what may be
declared. `DEFAULT_DRIVER` and `DRIVER_CHOICES` now live in
`strands_robots.registry`, which the seam already reads; they are no longer
importable from `strands_robots.drivers`.
