### Fixed: a `user_robots.json` key that no lookup can reach is refused instead of silently ignored

Every registry reader folds its query with `normalize_robot_name` (lowercase,
trimmed, dashes as underscores) before looking it up. `register_robot` folds the
name before it writes, but a hand-written overlay was merged verbatim, so a key
such as `rover-001` or `My_Arm` loaded without complaint and then answered no
query at all - not even its own spelling, which is folded first. The loader now
refuses such a key when the registry loads, naming the overlay file and the
spelling to rename it to, and warns when that spelling is already taken by a
shipped robot (renaming would replace it). It is refused rather than folded
because folding could collapse two keys onto one entry and keep whichever merged
last. `register_robot`'s write-time check applies the same rule, so it cannot
persist into an overlay the next read refuses, and `unregister_robot` removes an
exact key match first, so the refused key can be removed by the spelling the
error quotes.
