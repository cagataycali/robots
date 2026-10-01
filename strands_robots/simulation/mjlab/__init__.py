"""mjlab backend: GPU-vectorized MuJoCo (MuJoCo-Warp via mjlab) behind the SimEngine ABC.

``create_simulation("mjlab", num_envs=N)`` runs ``N`` copies of the world in
one process on one GPU. Env 0 answers the single-world ``SimEngine`` contract
(``get_observation``, ``send_action``, ``run_policy``), the ``*_batch``
methods expose every world at once.

Requires ``pip install 'strands-robots[sim-mjlab]'`` (mjlab pins
``mujoco~=3.11`` and ``torch>=2.14``).
"""

from strands_robots.simulation.mjlab.simulation import MjlabEngine

__all__ = ["MjlabEngine"]
