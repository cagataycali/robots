### Fixed: an on-policy gradient-norm clip the update cannot honor is now refused

`RLTrainSpec.max_grad_norm` is the last thing that touches a gradient before PPO
steps, and nothing judged it. `torch.nn.utils.clip_grad_norm_` does not either:
it scales every gradient by `max_norm / total_norm` whenever that ratio is below
one, and that expression is defined for values no caller can have meant. Two of
them were honored silently, on a run that reported `success` and wrote a
deployable checkpoint:

- `max_grad_norm=0` scales every gradient to **zero**, so the optimizer steps
  with no information. Measured over a seeded 60-step run, the resulting
  parameters were bit-identical to a never-trained control - a parameter delta of
  exactly `0.0000000000`.
- A **negative** bound negates the scaling ratio, so the update becomes gradient
  *ascent* on the loss: a parameter whose gradient is `[3.0, 4.0]` comes out of
  `clip_grad_norm_(-1.0)` as `[-0.6, -0.8]`, and the same seeded run moved its
  parameter sum to `17.8211606460` where the honored run moved it to
  `17.9833114612`, from a `17.9251941755865118` baseline.

`True` was a silent bound of one and `"1.0"` was silently coerced through
`float()`; `nan`, `None` and a list raised from inside `torch` mid-update, after
the environment, the networks and a full rollout had been built.

`PpoTrainer.validate` now reports the field through a shared
`gradient_clip_problems` gate, so `train()` fails closed before any rollout is
collected. The preflight stays read-only and reports rather than raises for every
value, including a real no float64 stands for (`10**400`) and a `numbers.Real`
registration with no working `__float__`. `inf` stays inside the domain: it is the field's only spelling of "do
not clip" and the consumer honors it by leaving every gradient untouched. Only
the on-policy backend clips, so FastSAC and the mock backend stay silent about a
field they never read.
