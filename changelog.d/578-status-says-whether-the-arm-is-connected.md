### Fixed: `status` on a real robot says whether the arm is connected

The reply now carries a `Connection:` line - `connected`, `not connected
(port '…' present)` or `not connected (port '…' not found on this machine)` -
read without touching the bus. Before, an unplugged arm answered `Robot
Status: IDLE` and an agent told the operator it was ready for instructions.
