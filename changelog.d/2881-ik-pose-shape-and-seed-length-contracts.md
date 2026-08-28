### Fixed: a wrong-shaped IK pose or seed is refused instead of breaking the consumer

`MinkIKBridge` documents a shape on three methods and checked it on none. `solve`
documents `target_pose` as a `(4, 4)` homogeneous pose and `q_init` as "length
`model.nq`"; `ee_pose` documents `qpos` the same way. Every wrong shape did
raise - the defect was *what* it raised.

Measured against a Franka Panda (`nq=9`), a pose that is not `(4, 4)` reached
`mink.SE3.from_matrix`, whose entire shape check is a bare `assert`, so the
caller received an `AssertionError` carrying **no message at all** - for a
`(3, 3)`, for a flat sixteen-vector and for a `(2, 4, 4)` batch alike. Under
`python -O`, where that assertion is stripped, the same three calls raised
`IndexError` ("index 3 is out of bounds for axis 1 with size 3") or `TypeError`
out of `mju_mat2Quat` instead, so the exception *type* depended on an
interpreter flag. A wrong-length `q_init` raised `ValueError: could not
broadcast input array from shape (6,) into shape (9,)` from the numpy
assignment inside `mink.Configuration.update`, naming neither the parameter nor
the class, and a length-one seed raised `mink.exceptions.InvalidTarget` - a
third type for one class of caller error. None of those is the `ValueError`
channel these methods document.

`solve_trajectory` meanwhile already refused a wrong-shaped `poses` batch **by
name** (`poses must be [N, 4, 4]; got (1, 3, 3)`) and then delegated to `solve`
per waypoint, so one wrong pose was reported two entirely different ways
depending on which entry point a caller used.

Each documented contract now has exactly one owner. `solve` refuses a pose that
is not `(4, 4)` and holds `q_init` to `model.nq` through `pose_vector_error`,
the shared fixed-length domain, which also subsumes the per-component
finiteness check it replaces. `ee_pose` holds `qpos` to the same length.
`solve_trajectory` and `tracking_error` inherit both refusals, because they
solve and read forward through those two methods - putting the checks in the
callers instead would leave a direct `solve` or `ee_pose` caller unguarded.
`tracking_error`'s `qpos_traj` is a caller's own array, and every row of it is
read back through `ee_pose`.

Two scope decisions are measured rather than assumed. The pose's shape cannot be
delegated to the shared fixed-length domain: `pose_vector_error(..., 16)`
accepts a `(2, 8)` array, because sixteen components is what it counts and that
is not what `matrix[:3, :3]` / `matrix[:3, 3]` slicing needs, so the shape is a
local check and the component domain still owns the values. And `ee_pose` gets
the length **only**, while `solve` keeps the full per-component domain: a
non-finite seed in `ee_pose` is already a *visible* reading - twelve of the
returned pose's sixteen entries come back non-finite and `tracking_error`
reports `{"mean_mm": nan, "max_mm": nan}` - whereas in `solve` it is not. The
cost splits the same way. The per-component domain reads each element in Python,
which is 8.44 us against a 22.9 us `ee_pose`, 37% of the call it would guard,
and 0.25% of `solve`'s 3.4 ms; the length comparison is 0.095 us.

No shipped caller is affected: `move_to` builds its target as `np.eye(4)` and
seeds from `data.qpos`, and the VERA action path builds its seed from
`model.qpos0`, so every internal caller already constructed a correct shape.
