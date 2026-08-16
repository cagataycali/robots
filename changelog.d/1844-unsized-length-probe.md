### Fixed: a vector parameter with no readable length is reported, not raised

Every validator that accepts a vector first asks "how many components is this?",
and the `hasattr(value, "__len__")` then `len(value)` spelling is unsafe for a
value class the library receives routinely: a 0-d NumPy array (`np.array(0.5)`,
or the result of a reduction such as `np.mean(...)`) and a 0-d torch tensor both
declare `__len__` and then raise from it. The `hasattr` probe passed and the
`len()` call escaped with a bare `len() of unsized object` naming neither the
parameter nor the method.

Four surfaces that publish a no-raise contract were affected: the MuJoCo
agent-tool router (every one of its ten vector parameters, so `position`,
`force`, `gravity`, `orientation`, `color` and the rest escaped dispatch rather
than returning a structured error), both length probes on `get_world_point`'s
`pixels` (whose structural checks exist to keep that envelope), and
`mjpeg_frames`' `size` (documented to fail only as `ValueError` with an
actionable message). `send_action`'s ordered-vector form reported such a value
as "action vector has a non-numeric entry: iteration over a 0-d array" and now
names the two shapes an action may take.

The length is read through one shared helper, `utils.sequence_length`, which
answers "no readable length" for a 0-d array and for a plain scalar alike, and
correctly sized NumPy vectors are accepted exactly as before.
