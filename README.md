# WBCConfig value domain - measured evidence

Real MuJoCo `unitree_g1` (headless, `MUJOCO_GL=egl`), driven by the real SONIC
PD law: a stub ONNX session asks every one of the 15 driven joints for a
`+0.35 rad` offset off its nominal stance, and the real
`compute_targets` -> `pd_control` chain writes `data.ctrl`.

`main_facts.json` / `branch_facts.json` carry every number in the figure; each
records the tree it was measured in (they differ, so the before/after pair is
two trees and not two runs of one).

| file | what it is |
|------|-----------|
| `wbc_config_value_domain.png` | the composed figure |
| `main_honored.png` / `branch_honored.png` | the honored config on each tree - max pixel delta 1/255 over 10 of 668,800 px |
| `main_scale_zero.png` | main, `action_scale=0.0`: success, 5.7 Nm, sags to 0.309 m |
| `main_gains_negative.png` | main, `kps=[-150.0]*15`: success, 461.9 Nm, collapses to 0.186 m |
| `main_scale_nan.png` | main, `action_scale=nan`: error blaming the embodiment, 0.0 Nm commanded |
