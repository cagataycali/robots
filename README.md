# WBC torque shim: teardown released the gains but kept its registration

Measurements behind strands-labs/robots#2196.

`install_wbc_torque_control` acquires two things - torque gains on the driven G1
actuators, and a registration in `world._backend_state["action_controller"]`.
`WBCTorqueController.uninstall` released only the gains.

## Scripts
- `capture.py` - two sequential 2 s WBC balance rollouts on ONE world, real
  `GR00T-WholeBodyControl-Balance.onnx`, headless MuJoCo (EGL). Spies on
  `WBCTorqueController.apply` to record which controller ran and the actuator
  mode it wrote into. Renders each end state.
- `cells.py` - the five teardown paths, run against main, #2196's first head and
  the combined branch.
- `mutate.py` - six plausible regressions x (the new tests / the pre-existing
  `tests/policies/wbc` suite).
- `measure.py`, `compose.py` - the dispatch ledger probe and the figure.

## Headline measurements
| | rollout 1 | rollout 2 |
|---|---|---|
| main | 100/100 steps on TORQUE actuators | **100/100 steps on POSITION SERVOS** |
| fixed | 100/100 on TORQUE | 100/100 on TORQUE |

All four calls returned `status="success"`. Rollout 1 is byte-comparable across
the two trees (`max|delta| = 1/255`, 0.00% of pixels differ); the rollout-2
panels differ on 16.32%.

| teardown path | main | #2196 head | combined |
|---|---|---|---|
| manual `install` + `controller.uninstall()` | LEAKED | LEAKED | released |
| auto hook cleanup | LEAKED | released | released |
| ...even when restoring gains raises | LEAKED | released | released |
| spares a controller registered since (manual) | released | released | released |
| spares a controller registered since (auto) | released | released | released |
