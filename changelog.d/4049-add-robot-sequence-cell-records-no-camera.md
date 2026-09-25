### Tests: the add_robot sequence cell records no camera

`tests/simulation/mujoco/test_add_robot_refuses_during_a_recording.py`'s
sequence cell - stop_recording, add_robot, start_recording again - asserts
the second dataset's `observation.state` width and values and reads no
pixel, yet both of its recordings took the default camera set, so the
world's free camera was rendered through OSMesa at every control step of
two rollouts and encoded to video at each stop. Both `start_recording`
calls now pass `cameras=[]`, the shape the episode-contract and
collection-loop cells already use; the cell went from 22.8 s to 1.4 s
locally and was the largest one left in the suite on the last measured CI
log. The two refusal cells beside it keep the default, since one of them
is the camera-only schema the refusal describes (#4049, towards #3869).
