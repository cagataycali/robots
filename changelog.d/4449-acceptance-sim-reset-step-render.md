### Tests: acceptance check 1, sim reset / step / render moves a real SO-101

`tests_integ/acceptance/` gains 1.0 acceptance check 1 (#3818): `Robot("so101", mode="sim")` loads the shipped SO-101 MJCF, a shoulder-lift command followed by `step(200)` reaches its target within 0.05 rad and visibly changes the 480x640 camera frame, `render()` returns a 640x480 PNG, and `reset()` puts the joint back where it started. Real MuJoCo, no doubles; run it with `pytest tests_integ/acceptance`.
