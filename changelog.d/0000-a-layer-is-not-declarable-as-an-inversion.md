### Fixed: an inversion the layer map invented is not declarable

`scripts/check_import_layers.py` graded two properties - no runtime import cycle,
and every upward import declared in a roster - and neither asked whether a
declared inversion deserved to exist. A member placed below the layers it reads
reports each of those reads as an inversion, so declaring them records the
placement rather than a dependency: `strands_robots/__main__.py` sat in `app`,
under `tools` and `dashboard`, which made the console entry point naming the
dashboard CLI it exists to start an `app -> dashboard` inversion with a roster
line of its own.

The script now also fails on a member whose declared layer is below the highest
layer it imports, when nothing that imports that member sits at or below there -
the move is available, so the inversion is not declarable. `__main__` moves to
the top layer, the one module nothing in the package can import, and its roster
line is deleted: six declared deferred inversions become five, and no layer
below the dashboard reaches into it at all. The three that remain
(`_hitl_audit`, `drivers`, `teleop_mixin`) are each forced by the code, which is
now measured rather than asserted.
