# harness issue #256 — verdict artifact

`list_objects()` does **not** drop cylinder objects. The 4 "missing" cylinders were
never created: the calls were refused by `add_object` because MuJoCo's documented
cylinder `size` is `[diameter, unused, full height]` (3 components) and the scene
script passed MJCF's `[radius, half-length]` (2).

Reproduce on Thor (`MUJOCO_GL=egl`, `PYTHONPATH=<robots checkout>`):

    python3 artifact.py     # builds both scenes, renders, writes artifact.json
    python3 compose.py      # composes verdict_2256.png (audits every number)
    python3 describe.py     # list_objects() lists box AND cylinder
    python3 agentpath.py    # direct vs sim(action=...) refusal are byte-identical

`artifact.py` asserts the reported 10-of-14 count reproduces exactly, that the
reporter's spelling compiles **0** cylinder geoms, and that the corrected
spelling compiles 4 and lists all 14.
