### Fixed: a LIBERO per-camera config the backend cannot accept is refused instead of silently dropping the camera

`LiberoAdapter(cameras=...)` documents each value as the keyword arguments
forwarded to `Simulation.add_camera`, so a key that method does not declare
cannot be honored on any call. The install loop is deliberately tolerant of a
sim *failing* to add a camera, and that tolerance covered this case too: a
one-character typo (`heigth`, `positon`, `resolution`) was logged at WARNING and
then treated as if the camera had been omitted, so the LIBERO policy's required
`image` / `wrist_image` view never entered the world and every subsequent
inference failed for a reason unrelated to the policy under test. A `name` key
collided with the name the install supplies from the mapping key, the same way.

The install now refuses such a config with a `ValueError` naming every unusable
key and the accepted set, before any camera is added, on the add path and on the
skip path alike - the skip path forwards the same mapping to the render-dimension
publisher, which reads `width` / `height` by name, so a misspelled dimension used
to publish the 256x256 fallback for a model-side camera under a successful
install. The accepted key set is read from the sim's own `add_camera` rather than
hard-coded, because MuJoCo and Newton declare `parent_body` and Isaac does not.
Every other `add_camera` failure stays best-effort exactly as documented.
