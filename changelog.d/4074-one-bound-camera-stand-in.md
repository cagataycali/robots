### Tests: the camera cells share one stand-in, bound to the real camera

Ten hand-rolled doubles across eight files stood in for the camera
`strands_robots.tools.lerobot_camera` opens, all modelling the same
`connect`/`disconnect`/`read`/`async_read` surface and none of them bound to it.
The factory itself was replaced with `lambda *a, **k: cam`, which accepts any
arity: dropping `rotation` from one `_create_camera(...)` call -- an argument the
real factory requires -- left all 471 camera cells green while the `record`
handler would have raised `TypeError: missing a required argument` against a
device.

Nothing else covers that seam. lerobot ships no `py.typed`, so every camera
symbol the tool imports is `Any` to mypy; how the tool calls a camera is pinned
by these cells or by nothing.

`tests/tools/_camera_stand_in.py` now holds one stand-in that takes its shape
from the symbols the module under test holds: each recorded call is bound
against the signature of the method it replaces on every camera the factory can
return -- the `Camera` contract it declares plus `OpenCVCamera` and
`RealSenseCamera` -- and the factory replacement is bound against
`_create_camera`'s own signature, so the geometry a handler opened is read
through the factory's parameter names instead of a positional slice.

`tests/tools/test_the_camera_tool_calls_a_camera_every_backend_accepts.py` reads
the 24 camera calls out of the module and binds each one, which reaches the call
sites no cell drives. Coverage of `tools/lerobot_camera.py` is unchanged at 98%
over the same 11 uncovered lines.
