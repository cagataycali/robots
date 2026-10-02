### Tests: acceptance check 8, the dashboard shows a live robot's joints byte-equal

`tests_integ/acceptance/` gains 1.0 acceptance check 8 (#3818): a `Robot("so101", mode="sim", mesh=True)` in its own process moves the shoulder lift and reports its joints; the real dashboard app on its own Zenoh session dials that peer on an explicit endpoint, and `GET /api/fleet` carries all six joint positions with the exact float the robot read. Real MuJoCo, real Zenoh, no doubles; run it with `pytest tests_integ/acceptance`.
