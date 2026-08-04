### Fixed: the run size a `TrainSpec` asks for is checked against one shared positive-count domain

`TrainSpec.steps` and `TrainSpec.global_batch_size` are the two factors of how
much training a spec asks for, and every supervised backend reads both straight
into a discrete consumer - `steps` bounds the optimizer loop (lerobot iterates
`range(step, cfg.steps)`) and `global_batch_size` becomes a `DataLoader` batch
size or a `--global_batch_size` flag. Four backends each carried their own
`if spec.steps <= 0` copy and none checked `global_batch_size` at all, so the
copies agreed with each other and shared one hole: a comparison admits every
value that is not comparably non-positive. `steps=True` validated clean and
trained for exactly one optimizer step; `steps=2.7` / `nan` / `inf` validated
clean and then raised `TypeError: 'float' object cannot be interpreted as an
integer` inside the backend's `range()`, after the dataset and the model were
already loaded; `steps="1000"` raised `TypeError: '<=' not supported between
instances of 'str' and 'int'` out of `Trainer.validate` itself, which is
documented to *return* problems. `global_batch_size` reached torch's
`DataLoader` unchecked for all of `0`, `-8`, `True`, `2.7`, `nan` and `"32"`.

`Trainer._run_size_problems` now routes both fields through the one shared
`positive_count_error` domain, and the four duplicated comparisons are gone.
The message names the backend that refused the value. The RL trainers are
deliberately off this path: per `TrainSpec`, a backend ignores the fields it
does not support, and they drive training from `total_timesteps` / `batch_size`
instead - reporting on a field they never read would be a false rejection.
