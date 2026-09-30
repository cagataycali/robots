### Fixed: Isaac cameras mount on a link with `parent_body`, and their frames have the size they were added with

`add_camera(parent_body=...)` was refused on Isaac, so every VLA wrist view (pi0.5
DROID `left_wrist_0_rgb`, LIBERO `image2`) was a world-fixed camera that did not
ride with the gripper - a view distribution none of those checkpoints was trained
on. The camera prim is now authored as a child of the link prim (a robot link as
`robot/link` or a bare name, or an absolute prim path, resolved like
`get_body_state`), with `position` and `target` in that body's frame and both
required, as on MuJoCo and Newton; an unknown body is refused with the robots' link
names. And `add_camera(width=224, height=224)` renders at 640x640 (the DLSS
ghosting floor), which every consumer received: `get_observation`, recordings and
datasets carried `[640, 640, 3]` where MuJoCo carries `[224, 224, 3]`. The camera now
keeps both sizes (`render_resolution` in the result), and every read-back is
resampled to the requested size (`INTER_AREA` colour, nearest-neighbour depth).
