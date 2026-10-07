### Fixed: `register_policy` stores the name it was given without surrounding whitespace

`register_policy(" myprov ", ...)` (or `"myprov\t"`, `"\nmyprov"`) stored the
padded spelling as the key, so `create_policy("myprov")` failed with a
did-you-mean hint quoting whitespace nobody can see. The name and every alias
are now stripped once before they are stored, and the built-in collision check
reads the stripped spelling, so `" mock "` no longer slips past it without
`overwrite=True`.
