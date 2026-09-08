### Bug Fixes

- **packaging**: the EarthRover native driver speaks HTTP to the vendor's
  earth-rovers-sdk - `GET /data` is every read, `POST /control` is what drives
  the rover - and no requirement named `requests`. The only bound the project
  stated lived in `[tool.hatch.envs.default].dependencies`, a hatch development
  environment that no install reaches. Measured against the lock, `requests` is
  absent from the base install and from 26 of the 32 other extras; the six that
  resolve to it reach it as a transitive of `lerobot` or `diffusers`, so
  `[all]` supplied the rover transport by way of an unrelated machine learning
  stack. A new `[earthrover]` extra declares it (folded into `[all]`: pure
  Python, no build step), and `connect_eagerly()` now names that extra instead
  of handing the reader a bare `pip install requests`. Pillow, msgpack and pyzmq
  were pinned in the same development-only place; they are declared in
  `[project.dependencies]`, `[groot-service]` and `[moveit2]`, so the duplicates
  are gone and `tests/test_earthrover_http_extra_is_declared.py` refuses the
  shape generally - a runtime distribution named only there is a second copy of
  a bound that can drift from the one an install resolves, with `hatch run test`
  green over the drift. No resolution changed: `uv lock` moves only the new
  extra's entries.
