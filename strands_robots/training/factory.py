"""Trainer factory - create_trainer() and runtime registration.

Mirrors ``strands_robots.policies.factory`` exactly, but resolves the
*training* class for a provider. A provider's trainer is declared in
``registry/policies.json`` under a ``"trainer"`` block alongside its policy::

    "lerobot_local": {
        "module": "strands_robots.policies.lerobot_local",
        "class": "LerobotLocalPolicy",
        "trainer": {
            "module": "strands_robots.training.lerobot",
            "class": "LerobotTrainer"
        },
        ...
    }

so a single provider name owns BOTH the inference class
(``create_policy("lerobot_local")``) and the training class
(``create_trainer("lerobot_local")``).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from strands_robots.registry.policies import get_policy_provider, list_policy_providers
from strands_robots.training.base import Trainer

logger = logging.getLogger(__name__)

# Runtime registration (for user-defined trainers not in JSON)
_runtime_registry: dict[str, Callable[[], type[Trainer]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_trainer(
    name: str,
    loader: Callable[[], type[Trainer]],
    aliases: list[str] | None = None,
) -> None:
    """Register a custom trainer provider at runtime.

    Use this to add training backends without editing policies.json.

    Example::

        from strands_robots.training import register_trainer

        register_trainer("my_vla", lambda: MyTrainer, aliases=["mv"])
        trainer = create_trainer("my_vla")

    Args:
        name: Provider name.
        loader: Zero-arg callable returning the :class:`Trainer` subclass
            (deferred import so heavy deps load only on use).
        aliases: Optional alternate names.
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_trainers() -> list[str]:
    """List provider names that have a trainer (JSON ``trainer`` block + runtime)."""
    names: list[str] = list(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    for provider in list_policy_providers():
        cfg = get_policy_provider(provider)
        if cfg and "trainer" in cfg:
            names.append(provider)
    return sorted(set(names))


def import_trainer_class(provider: str) -> type[Trainer]:
    """Import and return the :class:`Trainer` subclass for a provider.

    Resolution order:
      1. The provider's ``"trainer"`` block in policies.json.
      2. Auto-discovery fallback: ``strands_robots.training.<provider>`` with a
         class named ``<Provider>Trainer`` or the first ``Trainer`` subclass.

    Raises:
        ValueError: If no trainer can be resolved for the provider.
        ImportError: If the declared module can't be imported.
    """
    cfg = get_policy_provider(provider)
    if cfg and "trainer" in cfg:
        tcfg = cfg["trainer"]
        mod = importlib.import_module(tcfg["module"])
        cls: type[Trainer] = getattr(mod, tcfg["class"])
        return cls

    # Auto-discovery fallback: strands_robots.training.<provider>
    try:
        mod = importlib.import_module(f"strands_robots.training.{provider}")
        class_name = f"{provider.capitalize()}Trainer"
        if hasattr(mod, class_name):
            cls = getattr(mod, class_name)
            return cls
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, Trainer) and attr is not Trainer:
                return attr
    except ImportError:
        # No strands_robots.training.<provider> module; fall through to the
        # ValueError below so the caller gets the full "available trainers" list.
        pass

    raise ValueError(f"No trainer registered for provider '{provider}'. Available trainers: {list_trainers()}")


def create_trainer(provider: str, **kwargs: Any) -> Trainer:
    """Create a :class:`Trainer` for a policy provider.

    The training-side peer of ``create_policy``. The provider name is the SAME
    one used for inference, so ``create_policy("groot")`` and
    ``create_trainer("groot")`` address one family.

    Args:
        provider: Provider name or alias (``"lerobot_local"``, ``"groot"``,
            ``"cosmos3"``, or a runtime-registered name).
        **kwargs: Forwarded to the trainer constructor.

    Returns:
        A ready :class:`Trainer` instance.

    Raises:
        ValueError: If no trainer is registered for the provider.
    """
    # 1. Runtime registry first (user-registered trainers).
    resolved = _runtime_aliases.get(provider, provider)
    if resolved in _runtime_registry:
        return _runtime_registry[resolved]()(**kwargs)

    # 2. Registry / auto-discovery.
    TrainerClass = import_trainer_class(provider)
    return TrainerClass(**kwargs)
