### Removed: the `lerobot_calibrate` tool - calibrating is LeRobot's own CLI

`lerobot_calibrate` never calibrated anything. It listed, viewed, searched,
backed up, restored and "analyzed" the JSON files LeRobot writes under
`HF_LEROBOT_CALIBRATION`, and its own docs row said so. Recording a calibration
means disabling torque and moving one physical arm by hand, and LeRobot ships
that procedure as console scripts - `lerobot-find-port`, `lerobot-setup-motors`,
`lerobot-calibrate`, `lerobot-find-joint-limits`. A file manager over their
output is surface without essence, and it carried three defects of its own: it
restated LeRobot's calibration schema as loose `dict[str, int]` aliases instead
of importing `lerobot.motors.MotorCalibration`; its `analyze` action reported
statistics over raw range widths, which mean nothing without the motor model's
resolution (1024 vs 4096 counts); and its top level called
`logging.basicConfig()` and created `./.strands_robots/.calibration_backups` in
the working directory, so merely reaching the tool reconfigured the importing
process's root logger and wrote to disk.

`docs/hardware/tools.md` gains a `## Calibration` section naming the four
LeRobot commands and where the JSON lands. `lerobot_teleoperate` and
`lerobot_train` already read that JSON through LeRobot and are unchanged; code
that needs to parse it should import `lerobot.motors.MotorCalibration`.

That last defect was not unique to the removed tool, so the remaining three tool
modules that called `logging.basicConfig()` at import - `lerobot_camera`,
`lerobot_teleoperate`, `lerobot_train` - no longer do. A library must not take
over the host application's root logger; each module already had its own
`logging.getLogger(__name__)`, which is what the log calls use. Pinned by
`tests/tools/test_tools_lazy_import.py`, which reads every tool module with
`ast` and refuses a top-level `basicConfig` outright, plus any top-level
`mkdir` beyond the two session stores it names (`lerobot_teleoperate` and
`lerobot_train` create `.strands_robots/.sessions` under `Path.cwd()`; that is
load-bearing session state and is left alone, but a third one is refused).
