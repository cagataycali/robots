"""D5: one interface, the backends under it. Robot -> engine or driver -> transport -> motors.

Sources: strands_robots/robot.py (factory, mode, driver=, transport="twin"), strands_robots/simulation
(mujoco, newton, isaac backends), strands_robots/hardware_robot.py (lerobot driver),
strands_robots/drivers (native drivers and their transports), docs/hooks/robot_pages.py
LEROBOT_NETWORK_ADDRESS (which lerobot robots speak an IP, not a port).
"""

from excal import MUTED, Drawing

d = Drawing(
    "d05_driver_stack",
    "The same interface, get_observation, send_action, run_policy and the agent tool, sits on one "
    "of several backends: in simulation MuJoCo on the CPU, Newton or Isaac on a GPU; on hardware "
    "the lerobot driver or a native driver, each reaching the motors over a transport: a serial "
    "port, a TCP socket, DDS or ROS 2, or a MuJoCo twin for rehearsal.",
)

top = d.box(300, 30, 600, 70, "one interface", kind="accent", sub="get_observation, send_action, run_policy, the agent tool", size=18)

d.region(40, 150, 520, 200, 'mode="sim"')
mj = d.box(60, 200, 150, 70, "MuJoCo", sub="CPU, default", size=16)
nt = d.box(235, 200, 150, 70, "Newton", sub='GPU, backend="newton"', size=16)
isaac = d.box(410, 200, 130, 70, "Isaac Sim", sub="GPU, plugin", size=16)
d.text(60, 300, "the same registry entry and MJCF assets; never gated", size=13, color=MUTED)

d.region(620, 150, 560, 340, 'mode="real"')
lr = d.box(640, 200, 250, 70, "lerobot driver", kind="code", sub='driver="lerobot", the default', size=15)
nat = d.box(920, 200, 240, 70, "native driver", kind="code", sub='driver="strands"', size=15)

d.text(640, 300, "transport", size=13, color=MUTED)
ser = d.box(640, 320, 125, 56, "serial", sub="Feetech, Dynamixel", size=14, sub_size=12)
tcp = d.box(775, 320, 125, 56, "TCP", sub="Franka, Robotiq", size=14, sub_size=12)
dds = d.box(910, 320, 120, 56, "DDS", sub="Unitree", size=14, sub_size=12)
ros = d.box(1040, 320, 120, 56, "ROS 2", sub="any ROS stack", size=14, sub_size=12)
motors = d.box(640, 420, 520, 50, "motors", kind="chip", sub="the only thing the gate protects", size=14)

twin = d.box(60, 420, 480, 56, "twin", kind="chip", sub='transport="twin": the native driver steps the MuJoCo model', size=14)

for b in (mj, nt, isaac):
    d.path([(b["x"] + b["width"] / 2, 150), (b["x"] + b["width"] / 2, 200)])
d.path([(600, 100), (600, 130), (300, 130), (300, 150)])
d.path([(600, 130), (900, 130), (900, 150)])
d.path([(765, 150), (765, 200)])
d.path([(1040, 150), (1040, 200)])
for b in (ser, tcp, dds, ros):
    d.path([(b["x"] + b["width"] / 2, 376), (b["x"] + b["width"] / 2, 420)])
d.path([(765, 270), (765, 320)])
d.path([(1040, 270), (1040, 320)])

d.caption(40, 530, "the call does not change; what refuses it does: a missing extra, a busy bus, an operator who said no.")
d.save()
