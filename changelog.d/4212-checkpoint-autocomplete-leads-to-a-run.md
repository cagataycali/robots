### Changed: the checkpoint type-ahead is a combobox that knows the robot

The run form's checkpoint field already searched the Hub; it now ranks
checkpoints that name the selected robot first and marks them (the server takes
`robot=` on `/api/checkpoints/search`, matching `so101` against `so101`,
`so-101` and `so_101`, and a two-part name such as `unitree_go2` against its
parts), shows the policy-fit verdict on the first rows as an outlined pill
(fits / mismatch, nothing when the fit route has no evidence), and is a real
combobox: ArrowUp, ArrowDown, Home and End move the active row, Enter picks it,
Escape closes, with `aria-activedescendant` on the input so a screen reader
follows.
