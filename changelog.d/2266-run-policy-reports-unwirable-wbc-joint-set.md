### Fixed: `run_policy` reports an unwirable WBC joint set instead of raising past every caller

`run_policy(wbc_install_torque_control=True)` auto-installs the WBC torque shim
because a WBC policy driven through position servos diverges. The installer
gate is satisfied by *any* WBC joint on a position servo while
`install_wbc_torque_control` needs *every* one, so a humanoid model missing a
single joint reached the installer and its `RuntimeError` escaped `run_policy` --
a method documented to return a result dict. Callers reading `status` saw an
exception instead, and the raise landed before the rollout's `try`, so the
`finally` that restores the scene never ran.

The install is now reported as `status="error"` naming the joint that could not
be resolved, the actuator gains are left untouched and no controller is
registered, so the identical call succeeds once the model carries the joint.
