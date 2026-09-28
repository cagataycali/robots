### Tests: acceptance check 4, a recorded dataset re-reads in a fresh process

`tests_integ/acceptance/` opens with the first 1.0 acceptance check (#3818): `Robot("so101", mode="sim")` records a 10-second rollout, and an interpreter that never imports `strands_robots` reopens it with lerobot's `LeRobotDataset` and checks the episode and frame counts, the declared camera as a decodable 480x640 video feature, and a state column that moved. Real MuJoCo and lerobot, no doubles; run it with `pytest tests_integ/acceptance`.
