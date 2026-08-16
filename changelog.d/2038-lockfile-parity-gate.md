### Added: a lockfile that does not resolve the manifest is refused

Nothing compared `uv.lock` against `pyproject.toml` -- no workflow ran `uv lock
--check`, `uv sync --locked` or `uv sync --frozen` -- so the lock drifted for
fourteen days while CI stayed green. The lock was last written 2026-07-25 and the
manifest was edited three times after that with no relock, leaving `uv lock
--check` failing on `main`.

Because `uv.lock` is one of the manifests GitHub's dependency graph parses, the
drift was also a stale security surface rather than only a stale install: the
graph reported `lerobot 0.6.0`, the version `pyproject.toml` forbids (floor
`>=0.6.1`, bought by #1930 for exactly this guarantee), and carried no `roslibpy`
entry at all, so no advisory against it could be reported even though
`[rosbridge]` and `[all]` require it.

`.github/workflows/lockfile-parity.yml` now runs `uv lock --check` on the merge
commit, and `uv.lock` is relocked so the gate starts green: 14 packages added
(`roslibpy` and its transitive tree), 1 removed, and 2 upgraded -- `lerobot`
`0.6.0` -> `0.6.1` and `draccus` `0.10.0` -> `0.11.6`. The offline half of the
contract is pinned by `tests/test_lockfile_parity_gate.py`, so a locked version
below a declared floor, or a declared dependency missing from the lock, is
reported by the required check rather than only by the advisory gate.
