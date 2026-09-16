### Fixed: `serial_tool` and `pose_tool` without pyserial are refused with the install line

Both tools imported `serial` bare at their top, and pyserial is declared by no
extra of this project on its own (it arrives only inside `lerobot[feetech]`),
so on a core install `from strands_robots import serial_tool` answered
`No module named 'serial'` and nothing named the package to install. The two
modules now bind pyserial through `require_optional("serial",
pip_install="pyserial", ...)` like the Feetech driver beside them, so the
refusal - and the lazy loader's warning - reads `'serial' is required for the
Feetech serial bus tool (serial_tool) ... pip install pyserial`.
