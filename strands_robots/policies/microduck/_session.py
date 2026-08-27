"""Injectable ONNX session seam for the Microduck policy.

The provider talks to its ONNX weights through this narrow Protocol rather than
importing :mod:`onnxruntime` directly, so a test (or a non-CUDA host) can inject
a stub and exercise the whole obs-build / decode / last-action pipeline without
the runtime dependency. An :class:`onnxruntime.InferenceSession` already
satisfies it structurally - both members are present with these signatures.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelMeta(Protocol):
    """The slice of ``onnxruntime`` model metadata the provider auto-reads."""

    #: Self-describing config baked into the export: ``joint_names``,
    #: ``default_joint_pos``, ``action_scale``, ``command_names`` etc.
    custom_metadata_map: dict[str, str]


@runtime_checkable
class MicroduckSession(Protocol):
    """A minimal ONNX-inference seam: run the graph, and describe itself.

    :class:`onnxruntime.InferenceSession` satisfies this as-is. Stubs used in
    tests implement the same two members.
    """

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run the graph. ``input_feed`` maps input name -> ``[1, obs_dim]`` array."""
        ...

    def get_modelmeta(self) -> Any:
        """Return metadata whose ``custom_metadata_map`` self-describes the policy."""
        ...
