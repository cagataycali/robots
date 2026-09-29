"""A tiny ONNX actor stamped with the metadata mjlab's exporter writes.

Shared by the provider tests and the cross-provider contract graders so the
fixture is owned once: the actor is the identity on its first two observation
values, which keeps its outputs inspectable by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

JOINTS = ["a", "b"]


def write_actor(path: Path | str, terms: list[str], obs_dim: int, extra: dict | None = None) -> str:
    """Write the actor to ``path`` and return the path as a string."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((obs_dim, 2), dtype=np.float32)
    w[0, 0] = 1.0
    w[1, 1] = 1.0
    node = helper.make_node("MatMul", ["obs", "W"], ["actions"])
    graph = helper.make_graph(
        [node],
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    meta = {
        "observation_names": ",".join(terms),
        "joint_names": ",".join(JOINTS),
        "default_joint_pos": "0.100,-0.200",
        "action_scale": "0.500,0.250",
        "observation_terms_scale": ",".join(["1.000"] * len(terms)),
        "observation_terms_history_length": ",".join(["0.000"] * len(terms)),
        "observation_terms_clip": ",".join(["-inf;inf"] * len(terms)),
        "command_names": "twist",
    }
    meta.update(extra or {})
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(model, str(path))
    return str(path)
