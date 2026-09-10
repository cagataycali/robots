### Fixed
- `Trainer.export()` refuses a `checkpoint_dir` that is `None` or empty, naming the value. A failed `train()` returns `checkpoint_dir=None`, and the default export used to return that `None` as if an artifact had been produced.
