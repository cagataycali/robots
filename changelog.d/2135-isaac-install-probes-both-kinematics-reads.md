### Fixed: the LIBERO Isaac action-controller install now probes both kinematics reads

`LiberoAdapter._try_install_isaac_action_controller` probes at install time so
that, in its own words, "a broken kinematics read surfaces as a loud install
error at episode start (where the strict/non-strict policy applies) instead of
as one error envelope per action mid-eval". It probed only the Jacobian, and
translated only a `RuntimeError` from the getter.

`IsaacDeltaEEFController` reads two things per action: the Jacobian, and the arm
joint state via `get_observation`. The install validated the arm joints against
`robot_joint_names` (the articulation DOFs) instead, so an engine whose
observation omits or renames an arm joint, or reports a non-finite one, passed
the install with `_action_controller_error` unset and then failed every action -
the still-robot-with-a-green-eval shape the probe exists to prevent. A malformed
Jacobian payload also escaped as a bare `IndexError`, `ValueError` or
`KeyError`, outside the `_ControllerInstallError` contract the method documents,
so the caller's strict/non-strict policy never applied to it.

The probe now asserts exactly the preconditions the per-action solve re-checks -
the shape and finiteness of both reads - and every probe failure is a
`_ControllerInstallError`. A controller that installs can no longer fail the
solver on its first action.
