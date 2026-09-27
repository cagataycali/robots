### Security: the lockfile clears 47 of the 49 open Dependabot advisories

`uv.lock` pinned GitPython 3.1.50 (one CRITICAL, sixteen HIGH option-injection
and file-disclosure advisories), anyio 4.14.0 (a CRITICAL TLS host-name check
bypass), Pillow 12.2.0 (eleven HIGH heap and decompression-bomb advisories),
cryptography 49.0.0, aiohttp 3.14.1, mcp 1.28.0, setuptools 80.10.2 and an
autobahn 25.12.2 split - every one of them a transitive pin the resolver was
free to choose inside an advisory range, because nothing declared a floor.

The `Pillow` floor in `[project]` rises to 12.3.0 (it is a direct dependency),
and `[tool.uv] constraint-dependencies` gains one GHSA-annotated floor per
transitive package in the convention the file already uses: GitPython 3.1.59,
anyio 4.14.2, cryptography 50.0.0, aiohttp 3.14.3, mcp 1.28.1, autobahn 26.7.1,
setuptools 81.0.0. Relocked; `uv lock --check` and
`scripts/check_lockfile_parity.py` are clean; no source changes.

Two advisories remain open on purpose: `accelerate` (GHSA-4j2p-28q2-5m79) has
no patched release, and the `transformers` (5.10.0), `setuptools` (83.0.0) and
`torch` (2.13.0) fixes sit above the caps `lerobot 0.6.1` declares
(`transformers<5.6`, `setuptools<82`, `torch<2.12`) - they lift with the next
lerobot floor. The setuptools floor is written at the newest release that
resolves and says so.
