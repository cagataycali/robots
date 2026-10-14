"""A Policy turns an Observation into an Action.

Anything with ``act(obs, *, instruction=None)`` and ``reset()`` is a policy; a bare
``(obs) -> action`` callable (optionally taking ``instruction=``) is adapted by ``as_policy``.
"""

from __future__ import annotations

from .base import Policy, PolicyLike
from .callable import CallablePolicy, as_policy
from .requirements import Requirements
from .sequence import SequencePolicy

__all__ = ["Policy", "PolicyLike", "CallablePolicy", "SequencePolicy", "as_policy", "Requirements"]
