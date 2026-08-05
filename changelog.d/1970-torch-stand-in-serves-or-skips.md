### Fixed: the torch stand-in now skips with a reason instead of failing with an unactionable AttributeError

The numpy-backed torch stand-in the test suite installs when real torch is not
importable covers a subset of the surface, so a test can always reach past it --
directly, or through a package it imports, since `lerobot` reads `torch.dtype`
during import. That reach used to surface as `AttributeError: module 'torch' has
no attribute 'is_tensor'`, which names neither the stand-in nor the missing
dependency, so the first move was to debug the diff; at module scope it was a
collection error, and collection errors abort the whole run rather than one
module.

A reach now raises an exception that is both an `AttributeError` -- so every
`hasattr` probe and `except AttributeError` fallback behaves exactly as it does
against real torch -- and a pytest skip naming the attribute, what the stand-in
covers, and both remedies. Measured on a full torch-less run of `tests/`:
`pytest tests` went from aborting during collection without executing a single
test to running, and with `--continue-on-collection-errors` from 758 failed /
19704 passed / 281 skipped / 23 errors to 87 failed / 19744 passed / 977 skipped
/ 0 errors. Nothing that passed before fails now, and dunder lookup is left
alone so the import machinery and pytest introspection are untouched.

The discriminator for "is this torch real?" also has one home now,
`real_torch_installed()`, promoted out of a single test module; it cannot be
`pytest.importorskip("torch")`, because the stand-in registers a module in
`sys.modules` and the import therefore succeeds. The stand-in's docstring, the
conftest docstring and one test comment claimed it enabled "all unit tests"
without PyTorch, which was unverifiable by construction in CI (which always
installs torch); they now state the subset contract they can keep.
