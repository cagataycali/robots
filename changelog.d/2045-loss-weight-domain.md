### Fixed: the on-policy loss weights are refused when they cannot be honored

`RLTrainSpec.value_loss_coef` and `RLTrainSpec.entropy_coef` are the two scalars
that weight the terms of the objective PPO's update descends, and they are read
in exactly one place - the single expression that composes it:

```python
loss = surrogate_loss + spec.value_loss_coef * value_loss - spec.entropy_coef * entropy
```

Nothing judged either of them, and the multiplication cannot: it is defined for
values no caller can have meant, and every one of them reached the backward pass.
Measured on a seeded 60-step run whose honored checkpoint parameter sum is
`140.6023186540351162`:

- `entropy_coef=True` reported `success` and wrote a checkpoint whose parameter
  sum is `140.6158002523716277` - an entropy bonus at full weight where the field
  ships defaulting to `0.0`, requested by a value that reads as a flag. `bool` is
  an `int` subclass, so it lands as a coefficient of one.
- `nan`, `inf`, `-inf`, `"1.0"`, `None` and `[1.0]` raised **out of** `train()` -
  documented to return a terminal `TrainResult` and to fail closed on `validate()`
  first. A `nan` weight makes the loss `nan`, the optimizer writes `nan` into every
  parameter, and the *next* rollout samples the action distribution from them:
  `ValueError: Expected parameter loc ... of distribution Normal ... to satisfy the
  constraint Real()`, a torch message naming neither the field nor the value, after
  the env, the networks and a full rollout have been built.

Both are now reported by the read-only preflight, through a new
`loss_weight_problems` gate on the shared `finite_number_error` domain. The gate
bounds the *domain* rather than the floor: zero and negative stay accepted for
both fields, because both have a real reading - `entropy_coef=0.0` is the shipped
default, a negative entropy weight is a determinism penalty, and
`value_loss_coef=0` stops training the critic. Only the on-policy backend composes
this objective, so FastSAC and the mock trainer report nothing about either field.

`clip_param` and `init_noise_std` remain out of scope, and now for measured
reasons rather than by omission: `clip_param` is a clip *bound* whose `inf` is
coherently honored as "do not clip" (`clamp(ratio, -inf, inf)` returns `ratio`
unchanged), so it needs the endpoint decision the sibling `max_grad_norm` gate
records; and every non-finite `init_noise_std` is already refused by `torch`,
which rejects a `Normal` of non-positive or non-finite scale.
