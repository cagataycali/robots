### Fixed: the terrain example captures its frame after the robot stops moving

`examples/09_procedural_terrain.py` renders one still per terrain kind and its
whole value is that the still shows a robot resting on the heightfield. It
stepped a fixed 40 times first and called that settled ("Let the robot settle so
it rests on the heightfield (real contact)"), but a floating base spawns seated
with its feet just clear of the surface, so it drops and its undriven legs then
fold - which takes the better part of a second, not 0.08 s. Measured on the
shipped Go2, every captured frame was of a robot falling at 0.79-0.83 m/s,
216-249 mm above the height it settles to, and on `rough` with zero contacts at
all. How long that takes is a property of the model (the Go2 needs ~400 steps, a
longer-legged base thousands), so the settle is now measured from the base's own
reported speed and reported per terrain, `--steps` is the give-up bound, and a
base that never comes to rest is refused instead of photographed. The same
docstring's claim that the `examples/locomotion/` scripts run their WBC policy on
this terrain is dropped: all four build the flat default world.
