### Fixed: an unknown robot name says when `user_robots.json` is broken

A syntax error in a hand-edited `user_robots.json` (one trailing comma is
enough) hides every `register_robot` entry, and `Robot("<name>")` used to answer
with the generic "Unknown robot" refusal while the parse error went only to a
log warning. The refusal now names the overlay file and the decoder's line and
column. Lookups such as `get_robot()` still return nothing rather than raise on
a corrupt overlay.
