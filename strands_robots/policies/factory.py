"""Policy factory - create_policy() and runtime registration."""

import logging
import os
from collections.abc import Callable

from strands_robots.policies.base import Policy
from strands_robots.registry import import_policy_class, list_policy_providers, resolve_policy

logger = logging.getLogger(__name__)

#
# Runtime registration (for user-defined providers not in JSON)
#

_runtime_registry: dict[str, Callable[[], type[Policy]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_policy(
    name: str,
    loader: Callable[[], type[Policy]],
    aliases: list[str] | None = None,
):
    """Register a custom policy provider at runtime.

    Use this to add providers without editing policies.json.

    Example::

        from strands_robots.policies import register_policy

        register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
        policy = create_policy("my_provider", ...)
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_providers() -> list[str]:
    """List all available policy provider names (JSON + runtime)."""
    names = list_policy_providers()
    names.extend(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    return sorted(set(names))


class UntrustedRemoteCodeError(RuntimeError):
    """Raised when a HF model requires trust_remote_code but the user has not opted in."""


# Providers whose HuggingFace model loading path calls ``trust_remote_code=True``.
# Any provider that downloads and executes code from a model repository
# **must** be listed here so users are forced to explicitly opt in.
_HF_REMOTE_CODE_PROVIDERS: frozenset[str] = frozenset(
    {
        "lerobot_local",
    }
)


def _check_trust_remote_code(provider: str) -> None:
    """Enforce the trust-remote-code gate for HuggingFace-backed providers.

    Only providers listed in ``_HF_REMOTE_CODE_PROVIDERS`` are gated.
    These providers load models with ``trust_remote_code=True``, which
    allows **arbitrary code execution** from the model repository.

    Set the environment variable ``STRANDS_TRUST_REMOTE_CODE=1`` to opt in.
    """
    if provider not in _HF_REMOTE_CODE_PROVIDERS:
        return

    opted_in = os.environ.get("STRANDS_TRUST_REMOTE_CODE", "").strip()
    if opted_in in ("1", "true", "yes"):
        return

    raise UntrustedRemoteCodeError(
        f"Policy provider '{provider}' loads HuggingFace models with "
        f"trust_remote_code=True, which allows arbitrary code execution "
        f"from the model repository.\n\n"
        f"Only load models from organisations you trust.\n\n"
        f"To acknowledge this risk and proceed, set the environment variable:\n"
        f"    export STRANDS_TRUST_REMOTE_CODE=1\n"
    )


def _resolve_policy_class(provider: str, **kwargs) -> tuple[str, type[Policy], dict]:
    """Resolve ``provider`` to its policy class WITHOUT instantiating it.

    Imports the class and computes the effective constructor kwargs using the
    same three-stage lookup as :func:`create_policy` (runtime registry, smart
    string, then ``policies.json``), but never calls the constructor and never
    enforces the trust-remote-code gate. This lets callers inspect or run a
    class-level :meth:`Policy.preflight` check before paying the cost (and,
    for remote-code providers, the risk) of construction.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        ``(canonical_provider_name, PolicyClass, resolved_kwargs)``.

    Raises:
        ImportError / ValueError: Propagated from the underlying class import
            or smart-string resolution when the provider cannot be resolved.
    """
    # 1. Runtime registry (user-registered providers).
    resolved_name = _runtime_aliases.get(provider, provider)
    if resolved_name in _runtime_registry:
        return resolved_name, _runtime_registry[resolved_name](), dict(kwargs)

    # 2. Smart string (HF ID, URL, etc.).
    _needs_resolution = (
        "/" in provider
        or (":" in provider and not provider.replace("_", "").isalpha())
        or provider.startswith("ws://")
        or provider.startswith("grpc://")
        or provider.startswith("zmq://")
    )
    if _needs_resolution:
        try:
            resolved_provider, resolved_kwargs = resolve_policy(provider, **kwargs)
        except ImportError:
            resolved_provider = None
            resolved_kwargs = {}
        except Exception as e:
            logger.warning("Policy resolution failed for '%s': %s", provider, e)
            resolved_provider = None
            resolved_kwargs = {}
        if resolved_provider:
            return resolved_provider, import_policy_class(resolved_provider), dict(resolved_kwargs)

    # 3. Standard lookup from policies.json.
    return provider, import_policy_class(provider), dict(kwargs)


def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either a provider name or a smart string:

    - Provider name: ``create_policy("groot", port=5555)``
    - ZMQ URL: ``create_policy("zmq://localhost:5555")``
    - Shorthand: ``create_policy("mock")``

    All provider definitions live in ``registry/policies.json``.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        Policy instance ready for get_actions().

    Raises:
        UntrustedRemoteCodeError: If the provider loads HF models with
            ``trust_remote_code=True`` and ``STRANDS_TRUST_REMOTE_CODE``
            is not set.
    """
    canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    _check_trust_remote_code(canonical)
    return PolicyClass(**resolved_kwargs)


def preflight_policy(provider: str, observation_keys: set[str], **kwargs) -> None:
    """Run a provider's class-level :meth:`Policy.preflight` check, if any.

    Resolves ``provider`` to its policy class WITHOUT instantiating it (so no
    model weights are downloaded) and invokes the class's ``preflight`` hook
    with the runtime ``observation_keys`` and the provider kwargs. Providers
    that do not override :meth:`Policy.preflight` are a no-op.

    This is the fail-fast seam used by ``SimEngine.run_policy`` /
    ``eval_policy`` to catch a misconfiguration (e.g. sim camera names that
    cannot be routed to the model's declared image inputs) BEFORE the
    expensive ``create_policy`` download, instead of crashing deep inside the
    first inference. Resolution failures are swallowed (the matching error is
    surfaced authoritatively by the subsequent ``create_policy``); only the
    provider's own ``preflight`` ``ValueError`` propagates.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        observation_keys: Keys the runtime observation will contain (joint
            names + camera names).
        **kwargs: Provider-specific parameters (the policy_config).

    Raises:
        ValueError: When the resolved provider's ``preflight`` rejects the
            configuration.
    """
    try:
        _canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    except Exception as e:
        # Resolution problems (unknown provider, missing optional dep) are not
        # this hook's concern - create_policy raises the authoritative error.
        logger.debug("preflight_policy: could not resolve '%s' (%s); skipping", provider, e)
        return

    hook = getattr(PolicyClass, "preflight", None)
    base_hook = getattr(Policy.preflight, "__func__", Policy.preflight)
    if hook is None or getattr(hook, "__func__", hook) is base_hook:
        # Provider did not override the default no-op preflight.
        return
    PolicyClass.preflight(set(observation_keys), **resolved_kwargs)
