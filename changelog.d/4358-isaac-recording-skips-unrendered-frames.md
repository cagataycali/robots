### Fixed: a recorded Isaac episode does not open on a black camera frame

On Isaac, a camera's first frames of an episode can come back all zeros until
physics has stepped - render-only ticks do not light them - and the recorder
wrote them as they were: an so101 wrist camera recorded the first two frames of
every episode as black (MuJoCo: never), so a policy trained on the dataset saw
black inputs at every episode start. Until an episode has its first frame, a
frame whose declared camera image is missing or all zeros is now not written,
up to eight in a row (after which it is recorded as it is: that camera really
sees black). The episode starts at the first rendered frame, so it can be a
frame or two shorter than the rollout; frames after the first are always
written.
