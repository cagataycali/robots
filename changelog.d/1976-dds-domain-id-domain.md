### Fixed: a DDS domain id outside the RTPS port map is refused, not published process-wide

A ROS 2 / DDS domain id is an index into the RTPS port map. RTPS 2.2 sec. 9.6.1.1
derives every discovery port from it - `PB + DG * domain_id + d0` for the SPDP
multicast port, `PB + DG * domain_id + d1 + PG * participant_id` for the unicast
one - so with the standard parameters domain 232 lands on ports 65400/65410 and
domain 233 lands on 65650, past the end of the port space. Only an `int` in
`[0, 232]` names a domain, and the bound is the protocol's rather than a choice.

Five surfaces took one and none of them checked it: the hardware `Robot`'s
`ros2_domain`, a simulation backend's `ros2_domain`, and the `domain_id` of the
rclpy telemetry bridge, its hardware subclass, and the pure-RTPS bridge. Each
stored the value through a bare `int(...)`, which is where a `True` became
domain 1 and a `2.7` became domain 2 - a domain nobody named.

The rclpy bridge is why the range has to be checked at the boundary rather than
left to the transport. It pins the domain by writing `ROS_DOMAIN_ID` into the
process environment, and that write lands *before* `rclpy` is imported, so an
out-of-range value was never offered to DDS for rejection:

| `RosTelemetryBridge(domain_id=...)` | before | `ROS_DOMAIN_ID` after (started at `7`) | after |
| --- | --- | --- | --- |
| `0`, `5`, `232` | reaches the transport | `'0'`, `'5'`, `'232'` | unchanged |
| `233`, `300`, `2**31` | reaches the transport | **`'233'`, `'300'`, `'2147483648'`** | refused, `'7'` |
| `-1`, `-5` | reaches the transport | **`'-5'`** | refused, `'7'` |
| `True`, `False`, `2.7` | reaches the transport | **`'1'`, `'0'`, `'2'`** | refused, `'7'` |
| `nan`, `inf`, `None`, `[5]` | bare `ValueError` / `OverflowError` / `TypeError` | `'7'` | refused, `'7'` |

The environment write is process-wide and outlives the call that made it, so an
accepted out-of-range domain steered every later participant in the process -
and every subprocess that inherited the environment - at a domain nothing is
reachable on. It survived even when the construction went on to fail: the four
rows above that poisoned `ROS_DOMAIN_ID` all then raised `ImportError` for the
missing `rclpy`, leaving the bad domain behind them.

`utils.dds_domain_id_error` is now the one domain all five surfaces share, and
`utils.MAX_DDS_DOMAIN_ID` carries the port arithmetic that fixes its ceiling. It
lives beside `tcp_port_error` for the same reason that one does: the callers sit
in different layers, and the two transports exist to advertise the same topics,
so a domain the RTPS bridge cannot bind must not be one the rclpy bridge
publishes on. `bool` is rejected explicitly - it is an `int` subclass, so a bare
range test lets `True` through as a silent domain 1.

Each guard is placed ahead of its surface's transport probe. In the rclpy bridge
that means ahead of the `ROS_DOMAIN_ID` write, so a refused domain leaves the
environment as it found it; in the RTPS bridge it means ahead of the `cyclonedds`
probe, so the same caller mistake reports identically on an install with the
`[ros2]` extra and one without it. The now-redundant `int(...)` coercions are
gone, so an accepted domain is stored as the caller wrote it.

Every value that already named a domain still does: the existing usages across
the tests, docs and examples span 0-42 and none change.

Pinned by `tests/test_dds_domain_id_domain.py`: the RTPS port arithmetic is
asserted rather than the constant alone, a refused domain is shown not to touch
`ROS_DOMAIN_ID` while an accepted one still pins it, each guard is shown to
precede its transport probe, the five surfaces' verdicts are asserted equal over
the whole probe set, and a structural check requires every domain-taking surface
to either call the shared domain or forward the value to one that does - so a
sixth cannot ship without joining the rule.

`spin_period` / `poll_period`, the sibling knobs in two of those signatures, are
a different quantity with a different consequence (a wait budget: `0`, `-1` and
`nan` each return from `Event.wait` immediately and spin the command thread,
while `inf` raises `OverflowError` on it) and are left for their own change.
