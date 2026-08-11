# Artifact: the Isaac delta-EEF controller's numeric constructor domain

`capture.py` drives the real `IsaacDeltaEEFController` against a real MuJoCo
Panda. The controller reads current joint positions and the end-effector
Jacobian through injected callables, so a shim backed by a compiled MuJoCo
model runs the production conversion unchanged and gives its joint targets a
physical consequence -- Isaac Sim is not required to reach either constructor
decision.

Run it once per tree (`facts_main.json` / `facts_branch.json`), then
`compose.py` builds the figure and asserts every number it renders.
