# teleop auto-accept: a failed prompt answer was silent

`lerobot_teleoperate(action="start", auto_accept_calibration=True)` answers the child
process's calibration prompt by writing two newlines into its stdin from a background
thread. A write that failed was swallowed with a bare `pass`.

## Measured, both trees

| | upstream/main | this PR |
|---|---|---|
| start result `status` | `success` | `success` |
| reports "Session Started" | yes | yes |
| session store reports it running | yes | yes |
| newlines actually written | **0** | **0** |
| a record naming the failure | **none** | **WARNING** |

On main every caller-visible field for a FAILED write is byte-identical to a healthy
start, so the two outcomes are indistinguishable. A write that SUCCEEDS stays silent on
both trees, which is the posture `auto_accept_calibration` documents.

## Files

- `teleop_auto_accept.png` - the composed figure
- `facts-main.json`, `facts-pr.json` - the raw measurements, one per tree
- `scripts/capture.py` - runs both outcomes on whichever tree it is given
- `scripts/compose.py` - composes the figure and asserts every number it draws
- `scripts/mutate.py` - the 7-row mutation table (new module vs pre-existing arm)

Reproduce: `PYTHONPATH=<tree> python3 scripts/capture.py facts.json` in each tree, then
`python3 scripts/compose.py`.
