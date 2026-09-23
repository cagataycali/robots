### Fixed: `inverse_dynamics` reports one generalized force per DOF

`qfrc_inverse` is DOF-indexed, but the reported mapping was built one entry per
named joint, reading each joint's first DOF. A joint is not a DOF - a free joint
owns six, a ball joint three - so on every floating-base robot the whole of the
base wrench was reported as its first component under the joint's name, and the
remaining five were dropped with nothing said. Those are the components that
carry the load: a `unitree_g1` at rest reported 30 forces for `nv=35`, the base
entry holding `-4.7e-32` N while the vertical force holding the robot up
(`327.08` N, its weight) was absent; a `unitree_go2`, whose free joint is a bare
`<freejoint/>` and therefore unnamed, reported 12 for `nv=18` and dropped the
base wrench without even a key.

The mapping now carries one entry per DOF: a single-DOF joint keeps its bare
name (the spelling every hinge-only scene already reported, unchanged), a
multi-DOF joint's components are indexed `name[k]` from its first DOF, and a DOF
whose joint is unnamed is keyed `dof[i]`. The payload also reports
`dof_joint_names`, so the forces are labelled in DOF order on the same terms
`get_jacobian` names its columns and `get_mass_matrix` its diagonal.
