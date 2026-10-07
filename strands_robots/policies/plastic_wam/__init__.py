"""plastic-wam policy provider: a self-learning VLA (frozen Qwen3.5-4B System 2 + flow-matching System 1 with
test-time-training fast weights and a bounded, reversible plastic LoRA). Model code lives in the ``plastic_wam``
package (github.com/cagataycali/plastic-model); this provider adapts it to the strands-robots Policy API."""

from .policy import PlasticWAMPolicy

__all__ = ["PlasticWAMPolicy"]
