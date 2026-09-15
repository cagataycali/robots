### Fixed: a kinematically attached body is carried when the action controller owns stepping

`attach_bodies(mode="kinematic")` promises the child "follows the parent every
physics step", and one hook applies that follow after every `mj_step` the
backend issues. An `action_controller` declaring `owns_stepping` advances physics
inside its own `apply()`, and `_apply_sim_action` skipped its whole substep loop
- the re-pin included - to avoid double-stepping, so the child never followed:
it simply free-fell for the episode while `attach_bodies`, the controller and the
policy loop all reported success and the attachment registry still named it as
carried. Measured on a lift where the arm ends in the same pose either way, the
carried cube rose 155.6 mm on the default path and 0.8 mm - the rest-jitter floor
- under a stepping-owning controller, ending 265 mm from the gripper on the
platform it started on. `WBCTorqueController` declares the flag, and its own
module describes carrying the upper body of a `CompositePolicy`, so walking while
holding something reached this. The seam now re-pins after the controller's burst
returns, which is once per control step rather than once per substep; the carried
body is placed back before anything reads the scene.
