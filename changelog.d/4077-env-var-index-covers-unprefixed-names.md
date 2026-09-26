### Docs: seven environment variables the package defines now have a page

`docs/reference/configuration.md` promises every environment variable the
package reads, and the test that held it to that promise graded only the
`STRANDS_*` names - on the reasoning that anything else belongs to another tool.
That is true of `MUJOCO_GL` and `HF_TOKEN`, and false of every name this package
defines without the prefix, so seven knobs were read by the code and named on no
page: the Microduck driver's `MICRODUCK_MEDIA_SOCKET` and `MICRODUCK_TOF_SOCKET`,
the three SO-101 RTX timing knobs `SO101_RECORD_CONVERGE`, `SO101_IDLE_CONVERGE`
and `SO101_IDLE_RENDER_PERIOD`, the caller identity `DEVICE_CONNECT_CLIENT_ID`
that a device's RPC allowlist matches against, and `UNITREE_SDK_PATH`, the first
checkout the G1 tool reads a service's methods from.

All seven are documented, and ownership is now declared rather than inferred
from a prefix: a borrowed name is listed with the project that defines it, so a
variable added later is graded however it is spelled.
