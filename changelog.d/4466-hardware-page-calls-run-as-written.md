### Fixed: four hardware-page calls run as written

The LeKiwi sentence on the Feetech page now builds the laptop-side client,
`Robot("lekiwi_client", mode="real", remote_ip=...)`; the old
`Robot("lekiwi", mode="real", robot_ip=...)` raises since the native driver
became the hardware default. The microduck "Sim to real" sketch names
`policy_provider="microduck"` and its ONNX file, so it runs `MicroduckPolicy`
instead of `MockPolicy`. The Foxglove and calibration calls that only the
lerobot driver accepts (`foxglove=`, `id=`) now spell `driver="lerobot"`.
`tests/test_docs_real_mode_invocations.py` grades `Robot(..., mode="real")`
calls written inline in prose, not only in fences.
Every native-driver call those pages hand the reader now spells `port=`, which
the factory requires before construction, and the grader reports one that
omits it. The generated robot pages for robots with a native driver (SO-100,
SO-101, Koch, LeKiwi, HopeJR, Franka, UR, Open Duck Mini) no longer say lerobot
is the default: the native driver is, and the lerobot lines spell
`driver="lerobot"`.
