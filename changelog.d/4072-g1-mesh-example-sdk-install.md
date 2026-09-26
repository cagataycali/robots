### Fixed: the G1 mesh example installs the SDK its own refusal names

`examples/robots/neon.py` documented `pip install "strands-robots[mesh]" cyclonedds
unitree_sdk2py`. `unitree_sdk2py` is not on PyPI under that name, so the whole command ends
with `unitree-sdk2py was not found in the package registry` and nothing is installed. The
driver's missing-SDK refusal already carries the working recipe -- `[ros2]` for the
CycloneDDS binding plus the vendor checkout -- and the example now documents that.

The line was invisible to the examples grader because it wrapped: `pip install` ended one
source line and its arguments began the next, and the reader ended every command at the
newline, so the one unsatisfiable line under `examples/` resolved to no arguments at all. A
command now ends at its own code-span closer, across the wrap, and an install target the
library says pip cannot reach is refused.

The example's hardware note also conflated two absences. Without the SDK,
`connect_eagerly()` returns a named reason and the driver stays usable but unconnected; with
the SDK installed and no robot on the bus it returns `None` and `get_status()` reports
`connected`, because the DDS subscribers bind and simply never receive.
