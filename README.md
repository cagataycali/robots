# get_world_point: the camera-params read's failure arms

`SimEngine.get_world_point` makes two backend reads (`get_frame`, then
`get_camera_params`). Only the first read's failure was ever driven by the
suite. Reproduce on Thor, headless:

    MUJOCO_GL=egl PYTHONPATH=. python3 capture.py /tmp/art   # two real renders + facts.json
    MUJOCO_GL=egl PYTHONPATH=. python3 compose.py /tmp/art   # the figure (asserts every value)
    python3 mutate.py                                        # the 6x2 mutation table

`capture.py` loads the same orthographic scene the new MuJoCo test uses, so the
figure's left panel is the frame that test renders. `compose.py` re-derives
every number it draws from `facts.json` and refuses to save if any of them
disagrees. `mutate.py` scopes each anchor to its enclosing function, prints
in_fn vs in_file, and restores the source byte-identically in a `finally`.
