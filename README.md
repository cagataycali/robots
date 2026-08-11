# dataset-stack cells on an install without a simulator

`capture.py` runs the module both ways (mujoco present / blocked with
`nomujoco.py`), collects the nine-cell run/skip split and the mutation
outcome, and records one real MuJoCo dataset round trip. `compose.py`
asserts every number it draws against `measurements.json`.

    PYTHONPATH=. MUJOCO_GL=egl HF_HUB_OFFLINE=1 python3 capture.py
    PYTHONPATH=. python3 compose.py
