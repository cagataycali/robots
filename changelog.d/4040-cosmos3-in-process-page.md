### Docs: the Cosmos 3 page keeps the policy, the in-process backend gets its own page

`docs/policies/cosmos3.md` had grown to 2,022 words carrying two subjects: the
policy a caller builds over a WebSocket to the RoboLab server, and the
in-process `diffusers` backend, which does what only it can - return the
predicted world video and sound beside the action chunk, expose Cosmos 3's
`forward_dynamics` / `inverse_dynamics` modes, and hand back a raw
quantile-normalized end-effector delta that needs de-normalize -> decode -> IK
before MuJoCo can be driven with it.

The page is split at its `## Backends` table. `policies/cosmos3.md` (850 words)
keeps the policy: install, starting the server, the parameter block and its
address/timeout refusals, the embodiments, the action spaces and the `run_policy`
rollout. `docs/policies/cosmos3-diffusers.md` (1,315 words) takes the backend:
the `cosmos3-diffusers` install and its diffusers floor, the raw-action-layout
and safety-checker notes, the world-video example, the action-mode table and the
sim IK bridge with its per-domain quantile stats.

No prose was rewritten - every sentence, table, fence, refusal and tracking
figure is the text that was there. The nav, the see-also links and the
`cosmos3_sim_rollout.py` comment point at the half each names, and the two
guards bound to a moved section read the page that now carries it.
