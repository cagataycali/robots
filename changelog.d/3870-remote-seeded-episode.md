### Fixed: a seeded episode is reproducible over a `PolicyServer`, not only in-process

`LerobotLocalPolicy.reset` discarded its per-episode seed (`del seed`), on the
grounds that `set_eval_seed` seeds the process upstream of the call. That holds
for an in-process rollout and not for a served one: a lerobot policy draws its
flow-matching / diffusion noise from the process-global torch RNG and exposes no
seed kwarg of its own, so when the policy runs behind
`strands_robots.inference.server`, the seed the client forwards through
`reset(seed)` is the only seeding that inference process ever gets - and
throwing it away left two `seed=0` episodes over the wire differing by 0.621 rad
on the first action chunk, where in-process they were bit-identical. `reset` now
applies the seed through `reseed_client_rngs`, the shared applier `Gr00tPolicy`
and `Cosmos3Policy` already route their reset through for the same reason, so
the same episodes are bit-identical whichever side of the socket the policy is
on. In-process the reseed is redundant with the runner's own
`set_eval_seed(episode_seed)` and lands on the same RNG state, so a local
rollout is unchanged.
