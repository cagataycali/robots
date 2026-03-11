"""Auto-resolve a HuggingFace model ID or shorthand to (provider, kwargs).

All resolution rules now live in ``registry/policies.json``.
This module is a thin backward-compatible wrapper around
``registry.resolve_policy_string()``.

Examples::

    >>> resolve_policy("lerobot/act_aloha_sim_transfer_cube_human")
    ("lerobot_local", {"pretrained_name_or_path": "lerobot/act_aloha_sim_transfer_cube_human"})

    >>> resolve_policy("mock")
    ("mock", {})

    >>> resolve_policy("localhost:8080")
    ("lerobot_async", {"server_address": "localhost:8080"})
"""

from typing import Any, Dict, Tuple

from strands_robots.registry import resolve_policy_string

__all__ = ["resolve_policy"]


def resolve_policy(
    policy: str,
    **extra_kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a policy string to ``(provider_name, provider_kwargs)``.

    Delegates entirely to ``registry.resolve_policy_string()``.
    Kept for backward compatibility — new code should use the registry directly.

    Args:
        policy: Smart string — HF model ID, URL, or provider name.
        **extra_kwargs: Merged into the returned kwargs dict.

    Returns:
        ``(provider_name, kwargs_dict)`` ready for ``create_policy()``.
    """
    return resolve_policy_string(policy, **extra_kwargs)
