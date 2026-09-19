### Changed: the roster and GL-gate graders walk the tree once per session

Two more whole-tree graders re-walked and re-parsed the repository for every
cell that asked the same question of it. `tests/test_whole_tree_graders_roster_is_complete.py`
derived the roster twice - once in its `derived` fixture and again inside the
`roster()` call its `roster` fixture made - and
`tests/test_mujoco_render_assertions_are_gl_gated.py` ran its survey of the
tests tree in each of four cells. The tree does not change during a pytest
session, so each module now walks it once and its cells read the held result
set: the derived module paths in the first, and the surveyed labels and line
numbers, frozen, in the second. What is held is that small derived set, never
the parsed trees. `roster()` is still the function under test - it is handed the
derivation already made and refuses any other root - and `survey()` keeps its
per-root signature for the planted-source cells. Same machine, same 82 tests:
77.1 s on `main` to 35.2 s. No rule, population, exemption or message changes.
Towards #3869.
