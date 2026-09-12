### Fixed: the three RL posture flags are checked, not read by truthiness

`normalize_obs`, `normalize_advantage` and `autotune_alpha` on `RLTrainSpec`
each select a posture - wrap both observation streams in
`EmpiricalNormalization` or feed them raw; standardize advantages per batch or
use them as computed; build a temperature optimizer or hold `alpha` at
`init_alpha` - and every RL backend read the ones it consumes by truthiness
while its `validate()` graded every numeric knob around them. So the spellings
a caller reaches for to opt out (`"false"`, `"no"`, `"0"`) selected the
affirmative branch - the normalizers or the temperature optimizer the caller
had declined - and `0` or `None` selected the negative one without being a
declared spelling of it. `validate()` reported nothing for any of them.

`autotune_alpha` also gates whether `alpha_lr` is read, and that gate read it
by truthiness too: `autotune_alpha="false", alpha_lr=-1.0` was refused as the
rate of an optimizer the caller had asked not to build, while the flag whose
misread selected that branch was named nowhere.

Each flag now takes the shared `boolean_flag_error` domain - the one
`TrainSpec.resume` and `streaming` already take - through a field-scoped
`Trainer` gate consulted by exactly the backends that read it: all three for
`normalize_obs`, PPO for `normalize_advantage`, FastSAC for `autotune_alpha`,
ahead of the `alpha_lr` check, which now reads the rate only once the flag is
a usable `True`. A backend that ignores a flag reports nothing about it, and
the reader set is derived from the tree so a backend that starts reading a
flag is graded on arrival.
