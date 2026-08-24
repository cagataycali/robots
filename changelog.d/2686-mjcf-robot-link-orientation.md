### Fixed: a robot link's rotation is read from whichever MJCF spelling declares it, instead of being reported as identity

`load_mjcf` read each `<body>`'s `pos` into `BodyDef.position` but read no rotation at
all, so `BodyDef.orientation` kept its identity default for every link it reported. A
robot whose links the model rotates was reported upright with the load reporting
success, and identity is a valid orientation, so no caller could tell a link the model
never rotates from one whose rotation was dropped -- half of one pose was read and half
was silently discarded.

The sibling reader in the same module, `load_mjcf_scene_objects`, already resolves all
five of MJCF's mutually exclusive spellings (`quat`, `euler`, `axisangle`, `xyaxes` and
`zaxis`) through `_parse_orientation`, under the model-global `<compiler angle>` and
`<compiler eulerseq>`. The robot-link walk now reads the rotation the same way, in the
parent's frame -- the frame `position` is already reported in, and the frame MuJoCo's
`body_quat` stores, so nested links resolve too. `<compiler>` is read from the spliced
model, so a robot inheriting `angle="radian"` from an `<include>` is not read as
degrees. A body declaring two spellings is refused, as MuJoCo refuses such a model
outright.

Graded against `mujoco.MjModel.body_quat` across the shipped asset corpus, agreement
rises from 5474 of 8502 links to 7692, with no verdict changes, no link moving away
from the compiler, and 3030 corrected values spanning 307 assets. The 810 links that
still differ are all non-unit `quat` declarations, which MuJoCo normalizes and
`_parse_orientation` reports as written; that choice is shared with the scene reader
rather than local to this one, so it is left unchanged here and pinned by a test so it
cannot move silently.
