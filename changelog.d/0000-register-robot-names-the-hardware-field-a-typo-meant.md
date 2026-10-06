### Fixed: `register_robot` names a `hardware` key nothing reads, and the field it probably meant

`register_robot(hardware={"driver": "strands", "lerobot_tpye": "koch"})` stored
the misspelled key without a word, and nothing ever read it, so the robot came
back with no `lerobot_type`. A key outside `driver`, `lerobot_type` and
`requires_lerobot_from_source` now logs a warning naming the close match
(`did you mean 'lerobot_type'?`) or listing the known fields, the way a
near-miss `category` already does. When the same typo leaves an asset-less
robot with no buildable declaration, the `ValueError` carries the hint too.
