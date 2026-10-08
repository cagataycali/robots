### Tests: lock-exclusion cells end when the other thread asks for the lock

Thirteen MuJoCo cells (scene mutations, frame readers, `get_robot_state`) held
`sim._lock` for a fixed 1 s and read "the other side waited" from the timeout,
so every passing cell sat out the whole second. A small lock wrapper,
`tests/simulation/mujoco/_contended_lock.py`, now wakes the holder the moment a
second thread asks for the lock. Same verdicts, same planted negatives; the
three files run in 10 s instead of 23 s.
