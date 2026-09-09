### Fixed

- **A hidden layer of width zero is no longer accepted as an architecture.**
  `RLTrainSpec.hidden_dims` decides the shape of every network a from-scratch RL
  run trains, and nothing judged it. All three RL backends (`ppo`, `fast_sac`,
  `fast_td3`) expand the field identically, once per network they build - the
  on-policy actor and critic, and off-policy the actor, its Polyak target and
  all four Q heads - and `nn.Linear` accepts a width of zero as a legal layer.
  `torch` only warns ("Initializing zero-element tensors is a no-op"), the layer
  emits an empty activation, and the layer after it emits its bias alone, so the
  network's output stops being a function of the observation.

  Measured over a full `train()` on each backend, `hidden_dims=(16, 0)` returned
  a bit-identical action for an all-zero observation and an all-`50` one, while
  `train()` reported `status="success"` with a real `actor_loss` and a real
  `actor_updates` count and exported `policy.pt` + `policy_meta.json` - a
  deployable checkpoint whose actor commands one fixed action in every state the
  robot can reach. Sweeping the joint observation over `[-5, 5]` rad, the
  honored run's action spans `0.227` while the severed run's is exactly constant
  at `0.0`. On hardware that is an arm driving to a single pose and staying
  there, reported as a trained policy.

  `validate()` now asks each width of the shared `positive_count_error` domain,
  because a width is consumed directly as a tensor dimension: an integral float
  raises `TypeError` inside `torch` rather than being coerced, and `bool` would
  otherwise pass a bare `< 1` test as a silent width of one. That domain also
  refuses `np.int64`, which is not an over-refusal even though `nn.Linear`
  accepts it - `save_checkpoint` writes `list(spec.hidden_dims)` to
  `policy_meta.json` and `json.dump` raises `TypeError: Object of type int64 is
  not JSON serializable`, so such a run trains to completion and then loses the
  whole run at the save.

  The **empty** sequence stays accepted: it is the honest spelling of a linear
  policy, and its action still varies with the observation. So the domain is per
  element rather than on the length, and a problem names the offending index
  (`hidden_dims[1] must be a positive integer, got 0.`). A value that is not a
  sequence of widths at all - a bare `int`, `None`, a `str`, or a one-shot
  generator, which is consumed by the first network built and leaves every later
  critic a different shape - is refused as a whole.
