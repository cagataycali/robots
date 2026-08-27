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

import contextlib
import importlib
import sys
from collections.abc import Iterator

import pytest

pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local import list_policy_types  # noqa: E402
from strands_robots.policies.lerobot_local.processor import (  # noqa: E402
    _register_policy_processor_steps,
)
from strands_robots.policies.lerobot_local.resolution import (  # noqa: E402
    resolve_policy_class_by_name,
)

# The module whose import is the only thing that registers VLA-JEPA's steps.
_VLA_JEPA_PROCESSOR_MODULE = "lerobot.policies.vla_jepa.processor_vla_jepa"

# The three bespoke postprocessor steps VLA-JEPA registers (see lerobot
# ``policies/vla_jepa/processor_vla_jepa.py``). A checkpoint's
# ``policy_postprocessor.json`` references these by registry name, so they must
# be present after the type's processor module is imported or pipeline load
# fails with ``KeyError: Processor step '...' not found in registry``.
#
# This is the set the assertions are ABOUT. It is deliberately not the set used
# to reset the registry: that one is read off the module (see
# ``_steps_owned_by``), because resetting a subset of what a module registers
# leaves the module un-importable rather than merely unregistered.
_VLA_JEPA_PROCESSOR_STEPS = (
    "vla_jepa_clip_actions",
    "vla_jepa_pre_snap_gripper",
    "vla_jepa_binarize_gripper",
)


def _steps_owned_by(module_name: str) -> dict[str, type]:
    """Every registry entry whose step class was defined in ``module_name``.

    Read off the live registry rather than listed, so a step lerobot adds,
    renames or moves is picked up with no edit here. That matters because the
    reset below has to cover ALL of them: a name it misses is a name the
    re-import cannot re-register.
    """
    from lerobot.processor import ProcessorStepRegistry

    importlib.import_module(module_name)
    return {
        name: cls
        for name, cls in ProcessorStepRegistry._registry.items()
        if getattr(cls, "__module__", None) == module_name
    }


@contextlib.contextmanager
def _forced_reimport(module_name: str) -> Iterator[dict[str, type]]:
    """Make ``module_name``'s registration side effect happen again, then undo it.

    ``@ProcessorStepRegistry.register`` runs at class-definition time, so the
    only way to observe a module registering its steps is to import it fresh:
    evict it from ``sys.modules`` AND clear the names it owns.

    Clearing the names is not optional and it is not partial. LeRobot's
    ``ProcessorStepRegistry.register`` raises ``ValueError: Processor step
    '<name>' is already registered`` for a duplicate, so an import that meets
    one surviving name dies at that decorator -- leaving every step declared
    after it unregistered, and leaving the module itself permanently
    un-importable for the rest of the process (each later attempt trips the
    same surviving name). That is a strictly worse state than the one the reset
    was trying to create, and it is invisible to the caller because the module
    is gone from ``sys.modules`` and the registry still looks populated.

    Both halves are restored on the way out, so a session that runs these tests
    is left exactly as it was found. ``test_sys_modules_removal_leaves_no_orphan``
    grades an unrestored removal that orphans a *patched* reference; nothing
    patches this module, so that rule reads a removal here as legal and the
    restoration has to be carried here.

    Yields:
        The ``{registry name: step class}`` mapping the module owned on entry,
        which is also the set the re-import is expected to restore.
    """
    from lerobot.processor import ProcessorStepRegistry

    owned = _steps_owned_by(module_name)
    saved_module = sys.modules.get(module_name)
    try:
        sys.modules.pop(module_name, None)
        for name in owned:
            ProcessorStepRegistry.unregister(name)
        yield owned
    finally:
        # Restore the ORIGINAL step classes, not whatever the re-import left.
        # A fresh import builds new class objects for the same names, so
        # "leave it as found" means putting the entries that were there back --
        # otherwise a reference another test already holds and the registry
        # entry would be two different classes with the same name.
        for name, cls in owned.items():
            if name in ProcessorStepRegistry._registry:
                ProcessorStepRegistry.unregister(name)
            ProcessorStepRegistry.register(name=name)(cls)
        if saved_module is not None:
            sys.modules[module_name] = saved_module


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
    postprocessor pipeline resolves its steps by name. Clear the steps first,
    then prove our call re-registers all three -- so a checkpoint whose
    ``policy_postprocessor.json`` lists ``vla_jepa_binarize_gripper`` loads
    instead of dying with ``KeyError`` from the registry.
    """
    _require_vla_jepa_registered()
    from lerobot.processor import ProcessorStepRegistry

    with _forced_reimport(_VLA_JEPA_PROCESSOR_MODULE) as owned:
        # Non-vacuity: the named steps must be among the ones this module owns,
        # or the assertions below grade nothing. A rename upstream fails here,
        # naming both sides, instead of passing on an empty intersection.
        missing = sorted(set(_VLA_JEPA_PROCESSOR_STEPS) - set(owned))
        assert not missing, (
            f"{_VLA_JEPA_PROCESSOR_MODULE} no longer registers {missing}; "
            f"it registers {sorted(owned)}. Update _VLA_JEPA_PROCESSOR_STEPS."
        )

        _register_policy_processor_steps("vla_jepa")

        for name in _VLA_JEPA_PROCESSOR_STEPS:
            step_cls = ProcessorStepRegistry.get(name)
            assert isinstance(step_cls, type), f"{name} did not register to a class"


def test_forced_reimport_leaves_the_registry_and_sys_modules_as_found() -> None:
    """The reset used above puts the world back, so a sibling test is unaffected.

    The registration test is the only place in the suite that evicts a lerobot
    module to re-run its side effect. If it leaked, every later test in the same
    process that touches VLA-JEPA would fail for a reason belonging to this
    file, so the restoration is pinned rather than assumed.
    """
    _require_vla_jepa_registered()
    from lerobot.processor import ProcessorStepRegistry

    before = dict(ProcessorStepRegistry._registry)
    before_module = sys.modules.get(_VLA_JEPA_PROCESSOR_MODULE)

    with _forced_reimport(_VLA_JEPA_PROCESSOR_MODULE) as owned:
        assert owned, "nothing owned -- the reset would be measuring nothing"
        # Premise: inside the block the steps really are gone, so the exit
        # restoration below is doing work rather than observing a no-op.
        assert not set(owned) & set(ProcessorStepRegistry._registry)
        _register_policy_processor_steps("vla_jepa")

    assert dict(ProcessorStepRegistry._registry) == before
    assert sys.modules.get(_VLA_JEPA_PROCESSOR_MODULE) is before_module
    # And the module is still importable: the surviving-name hazard below would
    # make this raise.
    importlib.import_module(_VLA_JEPA_PROCESSOR_MODULE)


def test_clearing_only_some_of_a_modules_steps_makes_it_unimportable() -> None:
    """Why the reset clears every step the module owns, not just the named ones.

    Pins the hazard as behaviour: evict the module but leave ONE of its steps
    registered, and the re-import dies on that step's decorator. The module is
    then absent from ``sys.modules`` and un-importable for the rest of the
    process, and every step declared after the survivor stays unregistered --
    which surfaces later as ``KeyError: Processor step '...' not found in
    registry``, naming a step rather than this cause.
    """
    _require_vla_jepa_registered()
    from lerobot.processor import ProcessorStepRegistry

    with _forced_reimport(_VLA_JEPA_PROCESSOR_MODULE) as owned:
        assert len(owned) > 1, "need at least two steps to leave one behind"
        survivor, survivor_cls = next(iter(owned.items()))
        ProcessorStepRegistry.register(name=survivor)(survivor_cls)

        with pytest.raises(ValueError, match=f"'{survivor}' is already registered"):
            importlib.import_module(_VLA_JEPA_PROCESSOR_MODULE)

        # Best-effort registration swallows that ValueError, so the caller is
        # handed a registry still missing every other step.
        _register_policy_processor_steps("vla_jepa")
        assert set(owned) - {survivor} - set(ProcessorStepRegistry._registry)


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
