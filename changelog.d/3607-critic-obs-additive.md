### Fixed: privileged critic keys are added to what the critic sees, not substituted for it

`SimEnv`'s asymmetric actor-critic contract is documented in four places as
additive - "the critic may additionally see privileged simulation-only keys",
"appended to the critic observation" - but the constructor read
`critic_obs_keys` as the critic's *whole* observation. Naming one privileged key
therefore cost the critic every actor key: the value/Q head was sized for, and
fed, a vector that no longer contained the state whose value it was estimating,
and training ran to completion reporting a loss either way. An explicit `[]` -
which is `RLTrainSpec.critic_obs_keys`' own default - produced a zero-width
critic observation.

The critic observation is now `actor_obs_keys` followed by each
`critic_obs_keys` entry not already among them. A repeat is dropped rather than
appended, because a second copy of a scalar the critic already holds is not
information and its width is stamped into a checkpoint as `num_critic_obs`. The
symmetric default (`critic_obs_keys` omitted) is unchanged.
