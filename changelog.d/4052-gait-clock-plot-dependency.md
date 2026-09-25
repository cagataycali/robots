### Fixed: the gait-clock plot example names the dependency no extra declares

`examples/wbc/wbc_g1_gait.py --plot-clock` is the mode the file advertises as
needing no checkpoint and no GPU, and the command `docs/policies/wbc_gait.md`
credits its figure to. It draws that PNG with `matplotlib`, which no extra of
this project declares, and the file documented no install line at all: in a venv
built from the two sibling scripts' own line
(`pip install "strands-robots[wbc,sim-mujoco]"`) it exited 1 on
`ModuleNotFoundError: No module named 'matplotlib'` before drawing anything.
Both modes now document the line they need, and the page names the dependency
where it points at the command.

`tests/test_examples_document_the_interface_they_have.py` graded only the half of
the obligation the manifest can install, which left a module no extra covers to
the header and checked nothing there. It now grades that half for every runnable
example, with an import-name roster so a declared dependency spelled differently
(`PIL`, `cv2`, `yaml`) is not read as an undeclared one.
