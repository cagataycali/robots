### Added: the Franka arms drive real hardware over FCI

`panda`, `fr3` and `fr3_v2` each had a MuJoCo asset and no real-mode path -
lerobot registers no Franka robot type, so `Robot("panda", mode="real")` refused
all three by name and `driver="strands"` had nothing to build.

`strands_robots.drivers.franka.FrankaDriver` drives all three over the Franka
Control Interface through the `panda-py` binding over libfranka
(`pip install panda-py`), and registers itself for them, so
`Robot("panda", mode="real", driver="strands", port="172.16.0.2")` builds a real
arm. libfranka owns the 1 kHz realtime loop, so the driver owns what surrounds
it: the state decode (`q`, `dq`, `tau_J` and the Franka Hand's width), the
cadence a telemetry consumer reads at, and the gates a motion command passes
before libfranka sees it.

Joint names come from each arm's *own* MuJoCo model rather than one shared
invention, because the three do not agree - a Panda's joints are
`joint1..joint7` while an FR3's are `fr3_joint1..fr3_joint7`. Read them off
`driver.joint_names`; that is what lets one action dict drive the simulated arm
and the real one. A joint-space command must name all seven joints: FCI commands
a whole configuration, so completing a partial dict from the arm's present pose
would turn "move joint4" into a seven-joint motion the caller never wrote.
