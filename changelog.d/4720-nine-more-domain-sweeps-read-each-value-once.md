### Tests: nine more domain suites read each probe value inside one cell per surface

The clip-encoder quality, the off-policy warmup threshold, the camera-name
list, the rollout target types, unhashable entity names, the ZMQ client
timeout, the `add_camera` pose rule, the dashboard auth durations and the
Holosoma config scalars gave every probe value its own cell. Each surface now
loops its probe table and names the failing value in the assertion; tests that
restated another test's verdict (a refusal also checked word for word, a
"surfaces agree" check implied by every surface refusing) are folded in. The
nine files go from 639 cells to 225 with the same package lines executed.
