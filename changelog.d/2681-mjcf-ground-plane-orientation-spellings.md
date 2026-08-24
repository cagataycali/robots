### Fixed: a robot's authored plane is no longer stripped as the ground when its rotation is spelled with euler, axisangle, zaxis or xyaxes

`SpecBuilder.attach_robot` deletes a plane geom from a robot MJCF when it reads as
the world's z=0 ground, so a scene does not end up with two overlapping floors.
That flatness test asked the geom for `quat` alone. MJCF spells one rotation five
ways and MuJoCo keeps `euler`, `axisangle`, `xyaxes` and `zaxis` in a slot separate
from `quat`, so a plane rotated by any of those four read as unrotated and an
authored wall, ramp or angled panel sitting at z=0 was silently deleted on attach.
`zaxis` is the idiomatic spelling for a plane, because a plane's normal is its
local z.

The orientation is now resolved through all five spellings, honouring the
model-global `<compiler angle>` (which defaults to degrees) and `<compiler
eulerseq>` readings the module already applies elsewhere, and normalising `zaxis`
and `axisangle` axes. The position and size rules are unchanged, a genuinely flat
plane is still stripped in every spelling, and no message or caller-visible error
moves.
