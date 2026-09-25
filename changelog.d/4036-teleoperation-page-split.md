### Docs: the teleoperation page keeps the devices, the control loop gets its own page

`docs/hardware/teleoperation.md` had grown to 2,152 words, and most of the
excess sat inside one section: the mixin API stated the justification of each
refusal next to the refusal itself - why an empty `names` list must not widen to
every attached leader, why a slew-refused frame is counted rather than clamped,
which line of `_teleop_loop` makes the detach order load-bearing.

That rationale went first (2,152 -> 1,902 words), and what remained was
contract, so the page is split at its H2s. `hardware/teleoperation.md`
(947 words) keeps the devices: the `Teleoperator()` factory, the `*_leader`
refusal that points at it, the registry table and the ten recipes. The loop that
applies their frames is now `docs/hardware/teleoperation-loop.md` (1,088 words):
the mixin method table, the `attach_teleop` / `teleoperate` / `detach_teleop`
parameter domains, what one tick merges, the per-joint `STRANDS_TELEOP_SLEW_ABS`
bound, and the table of which teleop/robot pairings are zero-config.

Every env var, refusal string and action key survives the cut. `robot-control.md`'s
`#mixin-api` link follows the section it names, and the new page has a nav row.
