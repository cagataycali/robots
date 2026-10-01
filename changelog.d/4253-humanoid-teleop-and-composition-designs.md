### Added: whole-body teleoperation seam and driver composition, designed and proved in simulation

`strands_robots.teleop` holds `PoseSource` / `Retarget` / `Encoder` protocols, a scripted
`MockPoseSource`, `JointMapRetarget` and a `WholeBodyTeleoperator` that duck-types a lerobot
`Teleoperator`, so `teleoperate()` drives a headset the way it drives a leader arm; its layouts pin
the G1 recording columns, including the 31-state / 66-action table the LeRobot humanoid dataset
uses. `strands_robots.drivers.composite.CompositeDriver` presents several parts (a body, grippers, a
camera) as one robot with one key space, per-part units, primary-first writes, an ordered stop with
an e-stop latch on a partial halt, and per-part status on reads. Two project pages describe the
designs; the tests prove them on the MuJoCo G1.
