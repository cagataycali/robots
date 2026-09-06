### Fixed: `IsaacConfig.stage_path` must be a USD prim path

Every prim the Isaac backend creates is addressed by a path interpolated from
two caller-supplied components - `f"{stage_path}/Robots/{name}"`, and the same
shape for `/Objects/` and `/Cameras/`. The `name` half was already refused on
the shared `entity_name_error` domain; `stage_path` had none, so the values that
half rejects by name were accepted through this one and the interpolated result
was recorded in the cleanup registry `destroy()` releases and counts:
`stage_path=None` recorded `None/Robots/arm`, `stage_path=7` recorded
`7/Robots/arm`, `"World"` recorded a relative path `get_body_state` routes to
the wrong branch, `"/World/"` recorded `/World//Robots/arm`, and `"/My World"`
was recorded verbatim although USD transcodes a prim name outside its identifier
alphabet, so the prim does not land at the recorded path.

`stage_path` is now validated on construction - and therefore through
`IsaacConfig(...)`, `from_kwargs()`, `dataclasses.replace()` and the
`IsaacSimulation(stage_path=...)` shortcut - as an absolute USD prim path with
at least one component, every component matching `[A-Za-z_][A-Za-z0-9_]*`. The
refusal names the field and quotes the path the value would have produced. This
widens one previously working spelling: a `PurePosixPath` rendered correctly but
is not a `str`, and admitting a non-`str` that happens to render correctly is
what admitted `None`, whose rendering is the literal text `None`.
