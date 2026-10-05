### Added: a KUKA LBR iiwa 14 native driver over FRI

`Robot("kuka_iiwa", mode="real")` now builds `KukaDriver`
(`strands_robots.drivers.kuka`) instead of refusing the arm: lerobot registers
no KUKA type, so it was simulation-only. The driver speaks the Fast Robot
Interface through lbr-stack's `pyfri` in a separate session process, because
`pyfri`'s `step()` holds the GIL while it waits for the controller. Writes are
refused unless the FRI session is `COMMANDING_ACTIVE` in `POSITION` mode with
normal safety and active drives; a target outside the iiwa 14 range or past its
joint speed in one control period is refused, and every FRI cycle moves the
commanded position by at most the joint speed times the sample time. A halt
holds the last commanded position. `state`, `run_policy`, `start_task` and
`stop` are wired. `pyfri` is not on PyPI: build it from source against
pybind11 2.13 or newer; the build with its vendored pybind11 2.11 reports joint
1 for all seven joints under numpy 2, and the driver refuses to connect to it.
