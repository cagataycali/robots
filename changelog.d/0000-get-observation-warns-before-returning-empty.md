### Fixed: `get_observation` on MuJoCo and Newton logs why it returned nothing

MuJoCo and Newton returned `{}` from `get_observation` with no log when there
was no world (never created, or after `destroy()`), no robot to resolve, or an
unknown `robot_name`, while `send_action` and `step` on the same engine answered
`status="error"`. Each of those branches now logs a WARNING naming the cause -
the known robots for a typo, the ambiguity for several robots - matching the
Isaac backend. The return value is still `{}`.
