### Fixed: refuse an on-policy trust-region half-width that cannot be honored

`RLTrainSpec.clip_param` is the half-width of the trust region PPO is named for,
read twice per mini-batch - once to clip the policy ratio and once to clip the
value loss - and nothing judged it. `torch.clamp` cannot: it is defined for every
unusable value, so each produced a finite, successful, deployable run whose
objective was not the configured one. `nan` was the sharpest: both clipped terms
become `nan`, so every reported loss reads `nan`, but the gradient of
`torch.max` flows to the *unclipped* branch because comparisons against `nan` are
false - the run descended the unclipped objective and its checkpoint came out
bit-identical to an unclipped one, with PPO's defining mechanism silently off. A
negative half-width inverted the clamp bounds into a constant and flipped the
reported surrogate loss's sign, `0` pinned the value clip to `old_values`, and
`True` was a silent half-width of one.

`PpoTrainer.validate` now reports a half-width that is not a positive real, so an
unusable value is refused before the environment, the networks or the optimizer
are built. Positive infinity stays inside the domain - it is the field's only
spelling of *do not clip*, and `clamp(ratio, -inf, inf)` honors it by returning
the ratio unchanged. That is the same endpoint, for the same reason, as the
sibling clip bound `max_grad_norm`, so the two now share one domain helper
instead of carrying a copy each. The off-policy backend, which clips no policy
ratio, stays silent about a field it never reads.
