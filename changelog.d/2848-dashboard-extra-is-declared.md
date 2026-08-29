### Added: the dashboard's server dependencies are installable, and the refusal names the extra

`[tool.hatch.build.targets.wheel]` packages `strands_robots` entire, so
`strands_robots/dashboard/` shipped in every install of `strands-robots` while
the four packages its modules import shipped in none of them. They were declared
only in `[tool.hatch.envs.default].dependencies`, which is a hatch development
environment and reaches no installed copy, and `server.py`, `auth.py` and
`record_api.py` import `fastapi` / `webauthn` at module top level. So from a PyPI
install, or from a plain `pip install -e .`:

```
python -m strands_robots dashboard
# => ModuleNotFoundError: No module named 'fastapi'
```

Two things made that worse than a missing dependency usually is. The error names
`fastapi`, not the thing the reader is missing, so it reads as a broken virtualenv
and sends them looking for the fault in their own environment. And the comment on
the dev-env block, plus this pull request's own install instruction, both named a
`strands-robots[dashboard]` extra that `[project.optional-dependencies]` did not
declare -- and `pip` exits 0 on an unknown extra while installing none of it, so
following the documented command reported success and then failed later, at a
point that no longer looks related to the install.

This declares the extra (`fastapi`, `uvicorn`, `webauthn`, `python-multipart`),
folds it into `all`, and deletes the four duplicated pins from the hatch dev
environment -- that environment already sets `features = ["all"]`, so it keeps
resolving them through the extra rather than through a second copy that could
drift from it. That was the end state the deleted `TODO` asked for.

The refusal is routed through `require_optionals` in
`strands_robots/dashboard/__init__.py`. Python executes a package's `__init__`
before any submodule, so one call there covers every module in the package, and
the caller is told which extra supplies the gap:

```
'fastapi', 'webauthn' are required for the operator web dashboard
(strands_robots.dashboard)
Install with:
  pip install 'strands-robots[dashboard]'
  pip install fastapi webauthn
```

`require_optionals` rather than `require_optional` because the failure being
reported is one absent extra, which means several absent modules at once;
raising on the first turns a single install into three round trips.
`python-multipart` is in the extra but not in the gate's list -- it backs
FastAPI's form parsing rather than being imported here, so it has no module name
worth naming in a refusal.

`tests/test_dependency_audit.py::test_written_install_hints_name_only_declared_extras`
was failing on this branch naming `pyproject.toml:601`, and
`::test_require_optional_call_sites_name_declared_extras` is what keeps the
`extra="dashboard"` argument above honest if the extra is ever renamed.
`docs/dashboard/quickstart.md` now installs it; `docs/dashboard/index.md` already
installed `[all]` and so is correct by the fold.

**The relock, disclosed.**

Declaring an extra is a manifest change, so `uv lock --check` refuses the old
lock and the relock is the "deliberate relock" the removed `TODO` anticipated.
Its diff is large and almost all of it is mechanical: uv encodes extras into
`resolution-markers`, so adding a 33rd extra rewrites that table wholesale. The
control run is the evidence that none of it is incidental drift -- relocking the
*unmodified* manifest with the same uv produces a zero-line diff.

The semantic delta is six packages and two moves:

| change | package | why |
| --- | --- | --- |
| added | `fastapi`, `webauthn` | the extra's own contents |
| added | `pyasn1`, `pyasn1-modules`, `py-ubjson` | transitive under `webauthn` / `autobahn` |
| moved | `cbor2` 5.9.0 -> 6.1.4 | `webauthn` 3.0.0 requires it; `cbor2`'s only other requirer is `autobahn`, which accepts the range |
| split | `autobahn` 26.7.1 -> 25.12.2 **and** 26.7.1 | marker-only, and the older branch is confined to PyPy-on-win32 markers. Every CPython target -- linux, darwin, win32 -- keeps 26.7.1 |

The `autobahn` split is called out because a second, older version entry reads
like a downgrade to the mesh transport stack and is not one: no supported
platform resolves to it. `cbor2` crossing a major boundary is the one move worth
a reviewer's attention, and it is reachable only through `webauthn`.
