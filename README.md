# Artifact: gating MuJoCo render-success assertions on the shared GL probe

- `gl_gate.png` - the figure embedded in the PR description.
- `capture.py` / `compose.py` - produce it; every number is measured, none typed by hand.
- `facts.json` - the measurement the figure is built from.
- `noglhost_plugin.py` - a pytest plugin that emulates a host with no usable offscreen
  GL context by forcing the MuJoCo backend's cached render probe negative. Run with
  `PYTHONPATH=<dir> pytest -p noglhost ...` to reproduce the pre-fix failures.
- `mutation_check.py` - reverts the production behaviour each module pins and re-runs
  only the retained GL-free test, to show the split did not gut the pin.
