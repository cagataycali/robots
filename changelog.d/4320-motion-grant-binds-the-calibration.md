### Fixed: a motion grant is bound to the calibration the operator approved it under

The one-shot grant an operator's yes leaves behind was keyed on the tool, the
action, the port, the instruction and the motion fields, and not on the
`calibration` `pose_tool` hands the gate next to them. The calibration decides
where a degree target puts the joint, and a bus given none commands the servo's
full rotation instead of the arm's measured travel, so a yes for `position=30`
under one arm's file was spendable by the same numbers under another file or
under no file, with no second prompt (f017, CWE-863). `grant_key` now carries
`calibration_identity`: the content hash of the file (a symlink or relative
spelling of the same file spends the same grant), the hash of an inline record,
or the explicit word `none`. `calibration` leads `DETAIL_FIELDS`, so the
dashboard's line names the file before the numbers read inside it, and a
`pose_tool` motion without one says `calibration=none (servo full rotation)`.
A structural test keeps every field `pose_tool` and `serial_tool` hand the gate
in the key or on the roster.
