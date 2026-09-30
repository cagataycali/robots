### Added: `sim_call`, the simulation tool over the mesh

A simulation peer answered seven verbs on the wire while the MuJoCo tool behind
it publishes 77 actions, so an agent in another process could move a joint but
not add a cube. `sim_call` is now an allowed mesh action: `sim_action` names a
published simulation action and `params` carries its arguments, validated by
`mesh.security.validate_command` (63 actions admitted, 14 refused on purpose
with the reason in the refusal: world replacing actions, viewer windows, peer
host paths, rollouts that ride `execute`/`start`/`stop`, dataset replay; params
that are peer host paths or egress switches are refused on every action).
`Mesh._dispatch` serves it through the simulation's own `__call__`, on the
Simulation peer or on a child robot peer with `robot_name` bound, so the
simulation's validation and refusal wording apply unchanged. A hardware peer
refuses with a sentence, the e-stop lockout refuses, the audit record names the
`sim_action`, and a render's PNG travels base64 under the camera topic's cap.
