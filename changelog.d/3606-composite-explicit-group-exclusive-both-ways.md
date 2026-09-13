### Fixed: a `CompositePolicy` joint group is exclusive whichever child declares it

`upper_joints` was enforced as exclusive - a lower policy commanding into it is
refused - but the mirror was not. With `lower_joints` declared and `upper_joints`
left default, the upper policy was routed with only the names the lower policy
*emitted on that tick* excluded, so any joint the caller had assigned to the
lower policy but that the lower policy did not command was filled from the upper
policy's chunk instead, with no error.

On the pairing the class exists for that is the humanoid's waist: with
`lower_joints=WBC_G1_LEG_WAIST_JOINTS` and a whole-body manipulation policy that
emits all 29 G1 joints, `waist_yaw` / `waist_roll` / `waist_pitch` carried the
manipulation policy's target whenever the balance controller was silent on them,
and the balance controller's target whenever it was not - one configuration, two
owners, chosen per tick by a chunk's contents, and neither outcome reported.

Both directions are now graded from what each child *emitted* rather than from
what survived routing, so the verdict cannot depend on the other child's chunk:
the child whose own group is defaulted may not command into its sibling's
explicit group, and the refusal names the contested joints, both children and
the remedy. The boundaries are unchanged - two defaulted groups are still
arbitrated by lower precedence (the caller declared no owner), and two explicit
groups are still disjoint by construction.
