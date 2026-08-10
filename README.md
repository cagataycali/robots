# remove_camera: a refused recompile reported as a completed removal

- `measure.py` - probes remove_camera's behaviour when `spec.recompile` is refused
  (registry / live spec / compiled model / render / delayed application).
- `capture.py` - renders the three states; run once per tree with
  `MUJOCO_GL=egl PYTHONPATH=<tree> python3 capture.py <outdir> <main|branch>`.
- `compose.py` - builds the figure; asserts the two runs came from different
  trees and re-derives every rendered number from the two JSON dumps.
- `facts_main.json` / `facts_branch.json` - the measurements the figure quotes.
