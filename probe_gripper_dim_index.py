"""Probe gripper_dim_index on decode_vera_delta_chunk_to_targets."""
import math
import pathlib
import numpy as np

import strands_robots.policies.vera.sim_ik as sim_ik

print("TREE:", pathlib.Path(sim_ik.__file__).parents[3])


class Bridge:
    """Minimal MinkIKBridge stand-in: q[:3] IS the EE position."""

    class _M:
        nq = 6

    model = _M()

    def ee_pose(self, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q, dtype=float)[:3]
        return T

    def solve(self, target, q):
        out = np.asarray(q, dtype=float).copy()
        out[:3] = target[:3, 3]
        return out


# chunk: [trans(3), rot(3), gripper(1)] = 7 columns, T=2
# make each column distinguishable
CHUNK = np.array(
    [
        [0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 0.90],
        [0.11, 0.21, 0.31, 0.04, 0.05, 0.06, 0.10],
    ]
)
Q0 = np.zeros(6)

VALUES = [
    ("-1 (documented: last)", -1),
    ("6 (explicit last)", 6),
    ("0 (trans-x column)", 0),
    ("-5 (out of range neg)", -5),
    ("-99", -99),
    ("nan", math.nan),
    ("inf", math.inf),
    ("True", True),
    ("2.7", 2.7),
    ("6.0 (integral float)", 6.0),
    ("99 (>= D)", 99),
    ("'6' (str)", "6"),
    ("None", None),
    ("[6] (list)", [6]),
    ("np.int64(6)", np.int64(6)),
]

print()
print(f"{'value':26s} {'outcome':12s} {'gripper column read':28s} detail")
print("-" * 118)
for label, v in VALUES:
    try:
        out = sim_ik.decode_vera_delta_chunk_to_targets(
            CHUNK, Bridge(), Q0, rotation_dim=3, has_gripper=True, gripper_dim_index=v
        )
        g = out.get("gripper")
        gl = None if g is None else [round(float(x), 4) for x in np.asarray(g).ravel()[:4]]
        # which source column does that correspond to?
        src = "?"
        for c in range(CHUNK.shape[1]):
            if gl is not None and len(gl) == 2 and np.allclose(gl, CHUNK[:, c]):
                src = f"col {c}"
        qp = np.asarray(out["qpos"])
        print(f"{label:26s} {'ACCEPTED':12s} {str(gl):28s} src={src} qpos[0][:3]={np.round(qp[0][:3],4).tolist()}")
    except Exception as e:  # noqa: BLE001 - classifying every outcome
        msg = str(e).replace("\n", " ")[:74]
        names = "gripper_dim_index" in msg
        print(f"{label:26s} {'RAISED':12s} {type(e).__name__:28s} names_param={names} :: {msg}")
