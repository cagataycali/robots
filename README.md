# start_recording preflight refusals - measurement

`capture.py` records every fact (run inside the branch checkout, `PYTHONPATH=.`):
the mutation matrix (8 mutations x 2 arms, each restored byte-identically), which
refusal lines a driver executed before vs after, the full-suite coverage of the
two backend modules, and the proof that the rollout-rate refusal is unreachable
on Newton and Isaac (`_active_rollout_rates() == {}`).

`compose.py` renders the figure and asserts every rendered number against
`facts.json` before saving; it also checks text placement and that the image
border is pure white.
