### Fixed: one faulting sensor no longer erases the capabilities advertised after it

`Mesh._build_presence` announces a `topics` list naming the extended telemetry a peer can
serve -- `pose`, `imu`, `odom`, `lidar`, `health`, `hand`, `map` -- and decides each entry
by probing a provider attribute on the robot (`_pose`, `_imu`, `_battery`, ...). On
hardware those are properties that read a live sensor bus rather than plain fields, so one
can fail while every other sensor stays readable, and a sensor-bus fault is not an
`AttributeError`: it propagates straight through `getattr(robot, name, None)`.

All seven probes shared a single `try ... except Exception: pass`, so the first fault
abandoned the rest of the survey, and the advertised set was a prefix of the truth cut at
that fault. Measured on a robot exposing all ten providers, with exactly one of them
faulting:

| faulting provider | advertised before | advertised after |
|---|---|---|
| none | pose, imu, odom, lidar, health, hand, map | unchanged |
| `_pose` | health | pose, imu, odom, lidar, health, hand, map |
| `_imu` | pose, health | pose, odom, lidar, health, hand, map |
| `_odom` | pose, imu, health | pose, imu, lidar, health, hand, map |
| `_lidar_summary` | pose, imu, odom, health | pose, imu, odom, lidar, health, hand, map |
| `_battery` | pose, imu, odom, lidar, health | pose, imu, odom, lidar, health, hand, map |
| `_hands` | pose, imu, odom, lidar, health | pose, imu, odom, lidar, health, map |
| `_map_info` | pose, imu, odom, lidar, health, hand | pose, imu, odom, lidar, health, hand |

`_pose` and `_lidar_summary` recover every topic because those two are backed by sibling
providers -- `_slam_pose` / `_odom_pose` and `_lidar_state` -- which the survey now reaches
past the fault. `_map_info` is surveyed last, so it is the one provider whose fault cost
nothing before and costs nothing now: the loss was positional, which is why no fixed
expectation could have caught it.

The capability list is the only place those topics are announced, so under-reporting is
not cosmetic: a subscriber never learns the peer serves data the readers would have
answered with. `_pose` is surveyed first, which makes the worst case the common one -- an
expressive head whose pose provider faults advertised `health` alone while still serving
its IMU.

Each provider is now read through `_sensor_present`, which guards one attribute at a time
and continues to that topic's sibling providers on a fault, so `pose` survives a faulting
`_pose` when `_slam_pose` or `_odom_pose` still answers. That is the granularity
`strands_robots.mesh.sensors` already reads these same attributes at: its readers guard
each one separately, so a faulting `_battery` costs the health payload its battery fields
and leaves `cpu_load`, `mem_pct`, `disk_free_gb` and `uptime_s` intact. `docs/mesh.md`
already specifies this contract for the sibling payload -- "Every section of a state
snapshot is optional" and "A probe that fails therefore names itself" -- and the
advertisement now honours it, naming the dropped provider on the debug log.

Nothing readable is newly claimed: a robot exposing no providers, and a robot whose every
provider faults, both still advertise `health` alone, and a provider answering `None`
still contributes no topic.

The two pre-existing pins covered only the extremes, where a shared guard and per-provider
guards are indistinguishable -- `tests/mesh/test_mesh.py` drives a robot whose every
attribute raises (`topics == ["health"]`) and `tests/mesh/test_deep_mesh.py` one whose
every sensor answers (all seven topics). The mixed robot is what separates them, and
`tests/mesh/test_presence_capability_probe_isolation.py` grades it, deriving the
topic-to-provider map from `_build_presence` itself so a topic that later gains or loses a
provider is graded on arrival.
