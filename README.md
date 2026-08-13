# export_xml output_path validation - measurement artifacts

`capture.py` drives the real MuJoCo `export_xml` sink once per destination shape and
dumps every outcome to JSON. It was run in a worktree at `upstream/main` (83cc5272)
and on the branch; each dump records the tree it imported so the two cannot be
confused. `compose.py` builds the figure from those two dumps and asserts every
number it draws, including that the reference render is byte-comparable across the
two trees and that main fails 6 of 6 destination shapes while the branch fails 0.

- `facts-main.json` / `facts-branch.json` - the raw measurements
- `export_xml_path_validation.png` - the composed figure
