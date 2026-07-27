"""Smoke coverage for lerobot's VLA-JEPA policy on the ``lerobot_local`` path.

VLA-JEPA (Qwen3-VL-2B + V-JEPA2 world model, ``lerobot.policies.vla_jepa``)
ships in lerobot >= 0.6. It has no bespoke strands-robots load-path helper --
the generic ``lerobot_local`` machinery is expected to carry it end to end:

* ``resolution.py`` walks every ``lerobot.policies.*`` subpackage, so
  VLA-JEPA's ``@PreTrainedConfig.register_subclass("vla_jepa")`` registers and
  the type string resolves to its concrete policy class with no per-policy code.
* ``processor.py`` imports ``processor_<type>`` for the resolved type, so
  VLA-JEPA's custom postprocessor steps (Clip / PreSnapGripper / BinarizeGripper)
  land in lerobot's ``ProcessorStepRegistry`` before a checkpoint's
  ``policy_postprocessor.json`` references them by name.

These tests pin that "generic path already carries a brand-new VLA" invariant
against lerobot drift. They are dependency-light: they exercise resolution,
config defaults, and processor-step registration WITHOUT downloading the
multi-GB Qwen3-VL-2B / V-JEPA2 checkpoints or importing the heavy modeling
graph -- the same scope boundary as ``test_molmoact2.py`` (end-to-end MuJoCo
inference is covered by hardware/e2e validation, not this unit smoke).

They read lerobot's live registry rather than hardcoding, so they hold across
lerobot versions and skip cleanly on a lerobot too old to ship ``vla_jepa``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local import list_policy_types  # noqa: E402
from strands_robots.policies.lerobot_local.processor import (  # noqa: E402
    _register_policy_processor_steps,
)
from strands_robots.policies.lerobot_local.resolution import (  # noqa: E402
    resolve_policy_class_by_name,
)

# The three bespoke postprocessor steps VLA-JEPA registers (see lerobot
# ``policies/vla_jepa/processor_vla_jepa.py``). A checkpoint's
# ``policy_postprocessor.json`` references these by registry name, so they must
# be present after the type's processor module is imported or pipeline load
# fails with ``KeyError: Processor step '...' not found in registry``.
_VLA_JEPA_PROCESSOR_STEPS = (
    "vla_jepa_clip_actions",
    "vla_jepa_pre_snap_gripper",
    "vla_jepa_binarize_gripper",
)


def _require_vla_jepa_registered() -> None:
    """Skip unless the installed lerobot registers the ``vla_jepa`` type.

    ``vla_jepa`` first ships in lerobot >= 0.6; an older lerobot resolves fine
    for ``act``/``diffusion`` but has no such subpackage, so gate on the live
    registry rather than a version string.
    """
    if "vla_jepa" not in list_policy_types():
        pytest.skip("installed lerobot does not register 'vla_jepa' (needs lerobot >= 0.6)")


def test_vla_jepa_is_discoverable() -> None:
    """``vla_jepa`` shows up in the discovery surface on lerobot >= 0.6."""
    _require_vla_jepa_registered()
    assert "vla_jepa" in list_policy_types()


def test_vla_jepa_resolves_to_concrete_policy_class() -> None:
    """The type string resolves to VLA-JEPA's concrete, instantiable policy class.

    Pins the generic resolver: no ``vla_jepa``-specific branch exists in
    ``resolution.py``, so a return here proves the ``pkgutil`` subpackage walk
    picked up a policy that did not exist when the resolver was written.
    """
    _require_vla_jepa_registered()
    cls = resolve_policy_class_by_name("vla_jepa")
    assert isinstance(cls, type), f"vla_jepa resolved to {cls!r}, not a class"
    # Concrete, not the abstract PreTrainedPolicy fallback.
    import inspect

    assert not inspect.isabstract(cls), "vla_jepa must resolve to a concrete policy class"
    assert cls.__name__ == "VLAJEPAPolicy"
    from lerobot.policies.pretrained import PreTrainedPolicy

    assert issubclass(cls, PreTrainedPolicy)


def test_vla_jepa_config_chunking_and_normalization() -> None:
    """VLA-JEPA's config exposes the known-good chunking + normalization contract.

    The action-queue depth ``lerobot_local`` reads off the loaded config is the
    config's ``n_action_steps`` (chunk size 7); pin the defaults so a silent
    upstream change to the emitted chunk length is caught. Also pins the
    normalization mapping the norm-stats fallback must honour: VISUAL=IDENTITY
    (raw pixels), STATE=MEAN_STD, ACTION=MIN_MAX (the MIN_MAX action stats the
    checkpoint must ship, flagged as a load-time risk point).
    """
    _require_vla_jepa_registered()
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    cfg = VLAJEPAConfig()
    assert cfg.chunk_size == 7
    assert cfg.n_action_steps == 7
    # n_action_steps must never exceed chunk_size (lerobot enforces this in
    # __post_init__); pin the invariant the queue depth depends on.
    assert cfg.n_action_steps <= cfg.chunk_size

    norm = {k: v.value if hasattr(v, "value") else v for k, v in cfg.normalization_mapping.items()}
    assert norm["VISUAL"] == "IDENTITY"
    assert norm["STATE"] == "MEAN_STD"
    assert norm["ACTION"] == "MIN_MAX"


def test_vla_jepa_processor_steps_register_via_strands_path() -> None:
    """strands' generic processor-step registration lands VLA-JEPA's custom steps.

    Regression teeth for the generic import: ``_register_policy_processor_steps``
    is the only thing that imports ``processor_vla_jepa`` before a checkpoint's
    postprocessor pipeline resolves its steps by name. Unregister the steps
    first, then prove our call re-registers all three -- so a checkpoint whose
    ``policy_postprocessor.json`` lists ``vla_jepa_binarize_gripper`` loads
    instead of dying with ``KeyError`` from the registry.
    """
    _require_vla_jepa_registered()
    import sys

    from lerobot.processor import ProcessorStepRegistry

    # Start from a clean slate so the assertion proves OUR call did the
    # registration, not a leftover import from an earlier test. The
    # ``@ProcessorStepRegistry.register`` decorators only re-run on a fresh
    # module import, so evict the cached module too -- otherwise
    # ``_register_policy_processor_steps`` re-imports a cached module (a no-op)
    # and the unregistered steps never come back.
    sys.modules.pop("lerobot.policies.vla_jepa.processor_vla_jepa", None)
    for name in _VLA_JEPA_PROCESSOR_STEPS:
        try:
            ProcessorStepRegistry.unregister(name)
        except (KeyError, ValueError):
            pass  # not registered yet -- fine

    _register_policy_processor_steps("vla_jepa")

    for name in _VLA_JEPA_PROCESSOR_STEPS:
        step_cls = ProcessorStepRegistry.get(name)
        assert isinstance(step_cls, type), f"{name} did not register to a class"


def test_vla_jepa_registered_type_resolution_is_well_behaved() -> None:
    """Resolution for ``vla_jepa`` never leaks a raw internal exception.

    ``resolve_policy_class_by_name`` contracts to return a concrete class or
    raise a clean ``ImportError`` (never a raw ``TypeError`` / ``RuntimeError``
    from a missing optional VLA dep). Pin that contract specifically for the
    newest VLA so a drift that makes it leak is caught at the vla_jepa slice.
    """
    _require_vla_jepa_registered()
    cls: type | None = None
    try:
        cls = resolve_policy_class_by_name("vla_jepa")
    except ImportError:
        pytest.skip("vla_jepa registered but concrete class not importable in this install")
    except BaseException as exc:  # noqa: BLE001 - the point is to catch a leak
        raise AssertionError(
            f"resolve_policy_class_by_name('vla_jepa') leaked {type(exc).__name__} ({exc}); "
            "the contract is a concrete class or a clean ImportError."
        ) from exc
    assert isinstance(cls, type)
