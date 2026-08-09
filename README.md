# PolicyRunner.run horizon-pair domain

`capture.py` runs one scripted so100 rollout per scenario in a given tree (its
first line prints which tree it resolved) and dumps `facts.json`; `compose.py`
reads the two dumps, asserts every number it renders, and writes the figure.

Reproduce:

    MUJOCO_GL=egl PYTHONPATH=<tree> python3 capture.py /tmp/art_<tree>
    python3 compose.py
