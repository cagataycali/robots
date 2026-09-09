### Fixed: a `CompositePolicy` child is never queried with an observation of no keys

`lower_obs_keys` / `upper_obs_keys` select the observation each child is
queried with, and the selection is by name. The names belong to whatever
produced the observation: a MuJoCo world names them after the robot's own
joints (`"1"`, `"1.vel"`), while a LeRobot dataset spells the same reading
`"observation.state"`. A subset written in the wrong namespace shares no
key at all, and the child was handed `{}` - on every tick, of every
episode, with the rollout reporting success.

Measured on a 60-tick `run_policy` rollout of `so101` composing a
state-reading lower child over joints 1-3 with an upper child over 4-6:
with `lower_obs_keys=["observation.state"]` the lower child was blind on
60 of 60 ticks and left the joints it owns at `[0.000, 0.030, 0.025]`
rad, against `[0.418, 0.347, 0.384]` from the same composite with the
subset spelled in the world's namespace - a 0.418 rad divergence on the
joints one child was supposed to be closing the loop on. Both rollouts
returned `status="success"`.

The class already refuses the mirror mistake on the action side: a
`lower_joints` group sharing no name with what the child emits raises,
naming the group, the emitted names and the fix, because "a child whose
whole action dict is dropped reaches no actuator". A child whose whole
observation is dropped acts on no reading, which is the same failure one
step earlier. `RobotRLEnv` states the rule for the observation side
directly - "Validate obs keys up front so a typo fails loudly here, not
mid-rollout" - and refuses an `actor_obs_keys` the engine does not
produce, listing the missing keys and the available ones.

`CompositePolicy._filter_obs` now refuses a subset that selects nothing
from a non-empty observation, naming the seat, the child, the configured
group and the observation's own keys so the caller can read off the
namespace the subset should have been written in. The refusal precedes
both children's inference, so no model runs on a misconfigured composite.

Two boundaries are deliberate, and each is pinned. A *partly* satisfied
subset is still honored: the child got keys it asked for, and whether it
can act on those is its own contract (children read by name). An
observation with no keys at all is left alone, for the reason the action
side leaves a child that commanded nothing alone - there is nothing for
the subset to have missed.

`tests/policies/test_composite_child_observation_is_not_empty.py` pins
both seats, that neither child is queried when the selection is empty,
the two boundaries, and the `run_policy` envelope - which before this
change reported `status="success"` for the blind rollout.
