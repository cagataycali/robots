### Fixed: a G1 lidar field the message does not carry is `None`, not a fault code or a zero-point cloud

`G1Driver._on_lidar_state` and `G1Driver._on_lidar_cloud` were the last two
Unitree telemetry decoders still reading their fields with typed `getattr`
defaults and a bare `int()` / `float()`, after `_on_lowstate` moved its IMU
vectors and layout id onto the shared coercers in `strands_robots.drivers.base`.
A typed default is a well-formed value, and every one of these was shaped like a
reading. Measured through the decoders:

| the message carries | before | after |
| --- | --- | --- |
| no `error_state` | `code=-1`, `code_text="-1 (unknown)"` | `None`, `None` |
| `error_state=False` | `code=0`, `code_text="False (OK)"` | `None`, `None` |
| `error_state="3"` | `code=3`, `code_text="'3'"` | `3`, `"3 (unknown)"` |
| `error_state=b"\x03"` | raises; the whole frame is dropped | `None`, rates kept |
| no `cloud_frequency` | `freq=0.0` | `None` |
| no `width` | `count=0`, `width=0` | `None`, `None` |
| no header at all | a zero-point cloud | every field `None` |
| `width="n/a"` | raises; the whole frame is dropped | `width=None`, `height` kept |

`-1` renders as a fault code, `0.0` on `cloud_frequency` is a unit that has
stopped scanning, and `int(False)` is `0`, the one code `ERR_CODES` renders as
`OK` - a healthy lidar fabricated from a flag. A `width` of `0` is a zero-point
cloud, which the summary's own docstring names as the shape of the fault `count`
exists to show. Both consuming verbs, `g1_lidar_state` and `g1_lidar_summary`,
already documented every field as "or `None`"; the writers made that unreachable
for any driver that had received a message.

`code_text` now renders the coerced `code` rather than the raw field, so the two
describe one reading; a code that is no reading has no text. `count` needs both
dimensions and is `None` when either is. An unreadable field costs that field
and not the frame, matching what `_on_lowstate` does for its vectors.

`tests/drivers/test_g1_lidar_fields_are_readings_too.py` pins the rows above,
that neither decoder needs its `except` for a frame of non-readings, and widens
the structural scan from `_on_lowstate` alone to every `_on_*` decoder on both
Unitree drivers - no typed `getattr` default, no bare `int()` / `float()` - so
the six already on the rule are passing controls and a decoder added later is
held to it.

Pure routing: `g1.py` loses one executable statement (a local the record now
computes inline).
