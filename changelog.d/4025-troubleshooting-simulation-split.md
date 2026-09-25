### Docs: the troubleshooting sheet keeps the host, simulation gets its own page

`docs/troubleshooting.md` had grown to 2,444 words, half of it root-cause
narrative inside the `Fix` cells of its Simulation table - the three standing
causes of the EGL `llvmpipe` fallback, the download verdict's pre-fix wording,
and why a looser `tol` cannot rescue an unreachable orientation. That prose
belongs to this log and the pull requests, not to a symptom sheet.

The rationale is gone and the page is split at its own H2. Install, hardware,
policies, recording, mesh and agent integration stay on `troubleshooting.md`;
MuJoCo rendering backends, asset fetches and the `add_robot` / `move_to`
refusals are now `docs/simulation/troubleshooting.md`. Both pages are under the
1,500-word budget, every symptom row and remedy survives, and the quickstart's
headless `MUJOCO_GL` pointer follows the row it names.
