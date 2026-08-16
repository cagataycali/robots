### Fixed: a latched external wrench belongs to the body it was applied to

`apply_force` latched its wrench in `qfrc_applied`, one world-wide
generalized-force vector, and zeroed the whole buffer on every call to keep a
repeated call from accumulating. That also revoked every wrench already latched
on every other body: a second `apply_force` on a different body silently
cancelled the first while both calls reported success, so a wind field, two
thrusters, a magnetic gripper holding two parts, or a per-object disturbance
sweep all quietly reduced to whichever body was named last.

The wrench is now latched in the target body's own `xfrc_applied` row. Replacing
one body's wrench cannot touch another's, so per-call idempotency is kept
without revoking anyone else. No slice of `qfrc_applied` could have served the
same purpose: a wrench on a body part way down a kinematic chain writes into its
ancestors' degrees of freedom too, so there is no part of that buffer belonging
to one body.

Latching per body is also the more faithful reading of the documented contract
that the wrench is "applied on every subsequent step". MuJoCo re-maps a
Cartesian wrench through the current configuration on every step, whereas a
generalized force frozen at call time stops describing the caller's world-frame
wrench as soon as the body moves. Measured against the same force re-mapped
every step, a two-hinge arm swinging under a constant 2 N load now tracks that
reference to 0.005 rad where the frozen latch drifted 0.218 rad.

`apply_force(body, force=[0, 0, 0])` still stops one body, and `reset()` still
clears every latched wrench in the world.
