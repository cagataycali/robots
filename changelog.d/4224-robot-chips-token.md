### Docs: robot pages carry their chips as one token

A generated robot page's family, joints, sim, real and driver chips are written as `{{robot_chips:<name>}}`, which `docs/hooks/robot_pages.py` expands at build from the same registry and coverage rows, instead of a line of inline `<span>` markup. The rendered HTML is unchanged; the source site total drops 46,418 -> 45,620 words and the ceiling is lowered to match.
