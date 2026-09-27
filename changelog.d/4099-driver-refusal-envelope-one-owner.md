### Quality: the driver refusal envelope has one owner, `drivers.base.refuse`

Thirteen drivers (Booster, Crazyflie, Dynamixel, EarthRover, Feetech, Franka,
G1, Go2, Microduck, Reachy Mini, Robotiq, UR, Yahboom M3 Pro) and the Unitree
handle helper each carried a private one-line copy of the helper that builds
`{"status": "error", "content": [{"text": reason}]}`. They now import
`strands_robots.drivers.base.refuse`, and `undeclared_verb_error` builds its
envelope through it. Every refusal a driver returns is unchanged, byte for byte.
