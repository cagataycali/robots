### Fixed: `[lerobot]` declares psutil, so the session tools can be imported

`lerobot_train` and `lerobot_teleoperate` launch a detached background process and
then answer `status` / `stop` / `list` about it. Both import `psutil` at module
scope -- directly, and again through `strands_robots.tools._process_stop`, which
owns whether a signalled process actually exited and whether a recorded pid still
names the process the record was written for. No extra declared it.

Measured against the checked-in lock, `pip install 'strands-robots[lerobot]'`
resolved to 129 packages with no psutil among them -- and that is the command
`lerobot_teleoperate` itself prints as the remedy for its own absent dependency,
so following it produced an environment where `from strands_robots.tools import
lerobot_train` raised `ModuleNotFoundError: No module named 'psutil'` at
`lerobot_train.py:27`, before the tool could reach a message of its own. The
package and the lazy `strands_robots.tools` namespace both imported; only the two
session tools did not.

It survived because psutil arrived as a transitive of `accelerate`, under the GPU
extras (`[molmoact2]`, `[smolvla]`, `[kimodo]`, `[cosmos3-diffusers]`), and CI
installs `[all]`, which folds three of those in -- so the suite was green over the
extra a user is told to install.

`[lerobot]` now declares `psutil>=6.0.0,<8.0.0`, which `[lerobot-async]`,
`[molmoact2]`, `[smolvla]` and `[all]` inherit. The floor is a capability floor:
6.0.0 is the first release publishing a manylinux aarch64 wheel (every 5.9.x is
Linux x86_64 only), so below it an arm64 robot host has to build the C extension
from source, and the API the session verbs call is present on both sides of that
line.

`tests/test_session_process_dependency_is_declared.py` pins the row and the
general rule behind it: every third-party module this package imports
unconditionally at module scope is either named by a declared requirement, or
recorded as a transitive with the requirement that supplies it. Reachability alone
would have passed here -- `[all]` reached psutil through `accelerate` -- so what
is graded is whether the manifest takes responsibility for the dependency.
