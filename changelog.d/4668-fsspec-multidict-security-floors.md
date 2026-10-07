### Security: fsspec is floored at 2026.6.0 (GHSA-27vj-qcqg-25rc) and multidict at 6.9.1 (GHSA-54p9-h82j-f925)

Neither package is a direct dependency: fsspec arrives under `datasets`,
`huggingface-hub` and `torch`, multidict under `aiohttp` and `yarl`. The lock
pinned fsspec 2026.2.0, whose `ReferenceFileSystem` renders a crafted reference
spec as a template and runs it, and multidict 6.7.1, whose C extension leaks a
reference on update paths. `[tool.uv] constraint-dependencies` gains one
GHSA-annotated floor per package and `uv lock` moves exactly those two entries
(2026.2.0 to 2026.6.0, 6.7.1 to 6.9.1); nothing else in the lock changes and
`uv lock --check` plus `scripts/check_lockfile_parity.py` are clean.
