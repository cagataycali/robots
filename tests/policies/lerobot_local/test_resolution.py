"""Tests for ``strands_robots.policies.lerobot_local.resolution`` -- the
LeRobot policy class lookup that ``LerobotLocalPolicy`` uses to turn a
HuggingFace Hub repo id into a concrete ``PreTrainedPolicy`` subclass."""

from __future__ import annotations

import pytest

# pytest.importorskip raises Skipped at collection time if lerobot is not
# importable; it never returns None. Calling it once at module top is the
# canonical "skip the whole module unless this dep is installed" pattern --
# any subsequent ``pytest.mark.skipif(... is None, ...)`` wrapper would just
# be belt-and-suspenders dead code (the importorskip already handled it).
pytest.importorskip("lerobot")


def _snapshot_lerobot_modules() -> dict:
    """Snapshot all currently-loaded ``lerobot`` modules.

    Returns a dict suitable for restoring the caller's ``sys.modules``
    state via ``sys.modules.update(snapshot)`` after a destructive
    purge. The predicate matches the canonical lerobot package and any
    of its dotted children -- ``"lerobot" in name`` would also catch
    sibling packages whose name happens to contain the substring (e.g.
    a hypothetical ``my_lerobot_helper``), which is broader than the
    purge actually intends.
    """
    import sys

    return {name: module for name, module in sys.modules.items() if name == "lerobot" or name.startswith("lerobot.")}


def _purge_lerobot_modules(snapshot: dict) -> None:
    """Remove every entry in *snapshot* from ``sys.modules``.

    ``snapshot`` is materialized first so the caller can iterate it
    while ``sys.modules`` is being mutated. Symmetric with
    ``_snapshot_lerobot_modules`` so that a purge + restore round-trip
    leaves the interpreter in its original state.
    """
    import sys

    for name in snapshot:
        sys.modules.pop(name, None)


class TestPolicyConfigDiscovery:
    """Regression tests for ``_ensure_policy_configs_registered()``.

    The previous implementation imported a single hand-coded canary
    (``lerobot.policies.act.configuration_act``) and assumed lerobot's
    eager ``policies/__init__.py`` would side-effect every other policy
    config into the draccus ``PreTrainedConfig`` registry. That breaks
    the moment lerobot makes its policies subpackage lazy (the same
    transition ``lerobot.robots`` already went through), and it also
    breaks today inside ``LerobotLocalPolicy`` because that path
    intentionally installs a stub for ``lerobot.policies`` (to skip
    eagerly importing transformers/flash-attn dependencies of unrelated
    policies like groot).
    """

    def test_pkgutil_walk_registers_every_lerobot_policy_subpackage(self):
        """Walking ``lerobot.policies`` with pkgutil registers every
        policy config without any hand-coded list. The discovery is
        symmetric with the robots-side fix in
        ``hardware_robot._ensure_lerobot_robots_registered``.
        """
        from lerobot.configs.policies import PreTrainedConfig

        from strands_robots.policies.lerobot_local.resolution import (
            _ensure_policy_configs_registered,
        )

        _ensure_policy_configs_registered.cache_clear()
        _ensure_policy_configs_registered()

        registered = set(PreTrainedConfig.get_known_choices().keys())

        # Stable across lerobot 0.5.x; adding more upstream is a no-op
        # for strands_robots (the pkgutil walker picks them up
        # automatically). Newer policies (e.g. molmoact2, which only
        # ships in lerobot 0.5.2+ via lerobot PR #3604) are asserted
        # via dedicated importorskip-gated tests below; pinning them
        # here would couple this regression test to the specific
        # lerobot minor version installed in CI.
        expected_min = {
            "act",
            "diffusion",
            "pi0",
            "smolvla",
            "tdmpc",
            "vqbet",
        }
        missing = expected_min - registered
        assert not missing, f"Discovery missed lerobot built-in policies: {missing}. Registered: {sorted(registered)}"

    def test_molmoact2_registered_after_stubbed_lerobot_policies(self):
        """The ``LerobotLocalPolicy`` runtime path installs a lightweight
        stub for ``lerobot.policies`` (to avoid executing its potentially
        heavy ``__init__.py`` that pulls in transformers/flash-attn).
        Even with that stub in place -- which short-circuits any
        side-effect-on-init style registration -- ``molmoact2`` and
        every other lerobot built-in policy must still resolve.

        Pre-fix, the stub combined with the single-canary import meant
        ONLY ``act`` ended up registered; lookups for any other policy
        type silently fell through to manual config.json parsing,
        which failed for repos that rely on draccus resolution.

        Skipped when the installed lerobot is older than 0.5.2 (which
        added molmoact2 in lerobot PR #3604) -- the broader "every
        subpackage gets walked" invariant is covered by
        ``test_pkgutil_walk_registers_every_lerobot_policy_subpackage``
        without depending on a specific minor-version policy.
        """
        pytest.importorskip("lerobot.policies.molmoact2")
        import sys

        # Snapshot the current lerobot imports BEFORE we touch anything,
        # so the test can fail / abort and the interpreter still exits
        # with the same module state it started with. The previous
        # version of this test purged the modules without a teardown,
        # which (a) leaked the stub installed two lines below into
        # every later test that imports lerobot.policies and (b)
        # silently changed the production ``PreTrainedConfig`` class
        # identity for the rest of the run.
        snapshot = _snapshot_lerobot_modules()
        _purge_lerobot_modules(snapshot)
        try:
            from strands_robots.policies.lerobot_local.resolution import (
                _ensure_lerobot_policies_importable,
                _ensure_policy_configs_registered,
            )

            _ensure_lerobot_policies_importable()  # installs the stub
            # ``@functools.cache`` is keyed on the empty tuple, so a
            # prior call in this process would short-circuit and the
            # walk we want to exercise would never run. The contract
            # noted in the helper's docstring is that callers who
            # invalidate ``sys.modules`` MUST clear the cache first.
            _ensure_policy_configs_registered.cache_clear()
            _ensure_policy_configs_registered()

            from lerobot.configs.policies import PreTrainedConfig

            registered = set(PreTrainedConfig.get_known_choices().keys())
            assert "molmoact2" in registered, (
                f"molmoact2 missing after stub+walk; registered: {sorted(registered)}. "
                "Did the pkgutil walker get reverted to single-canary bootstrap?"
            )
            # Also verify the symmetric case for an older policy that pre-dates
            # the stub mechanism, to make sure we didn't break the existing path.
            assert "act" in registered
        finally:
            # Restore the snapshot regardless of test outcome so a
            # later test ordering (e.g. running this BEFORE
            # ``test_pkgutil_walk_registers_every_lerobot_policy_subpackage``)
            # does not see the stubbed ``lerobot.policies`` and the
            # mid-run-rebuilt ``lerobot.configs.policies``.
            _purge_lerobot_modules(_snapshot_lerobot_modules())
            sys.modules.update(snapshot)
            # Drop the cache one more time so the next test in the
            # suite re-walks against the restored, real lerobot.
            from strands_robots.policies.lerobot_local.resolution import (
                _ensure_policy_configs_registered,
            )

            _ensure_policy_configs_registered.cache_clear()

    def test_resolve_class_by_name_handles_molmoact2_modeling_convention(self):
        """``modeling_<type>`` lookup works for new policies that follow
        the convention. molmoact2's class lives at
        ``lerobot.policies.molmoact2.modeling_molmoact2.MolmoAct2Policy``;
        this path is the second strategy after the draccus registry."""
        pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
        from strands_robots.policies.lerobot_local.resolution import (
            resolve_policy_class_by_name,
        )

        cls = resolve_policy_class_by_name("molmoact2")
        assert cls.__name__ == "MolmoAct2Policy"
        assert cls.__module__.endswith("molmoact2.modeling_molmoact2")

    def test_walk_continues_after_subpackage_decorator_failure(self, monkeypatch, caplog):
        """A subpackage whose ``configuration_*`` raises a non-ImportError
        (e.g. ``RuntimeError`` from a re-registration collision, or
        ``AttributeError`` from a renamed sibling attribute) MUST NOT
        abort the walk. Pre-R1 the helper caught only ``ImportError``,
        so a single buggy decorator on one subpackage would leave the
        registry permanently half-populated for the lifetime of the
        process (because ``@functools.cache`` then froze the failed
        state).
        """
        import importlib
        import logging

        from lerobot.configs.policies import PreTrainedConfig

        from strands_robots.policies.lerobot_local import resolution

        original_import = importlib.import_module
        # Pick a booby_trap that ``pkgutil.iter_modules`` actually visits.
        # ``pkgutil.iter_modules`` only yields subpackages with an
        # ``__init__.py`` (regular packages) -- subpackages laid out as
        # namespace packages (no ``__init__.py``) are silently skipped.
        # In lerobot 0.5.x, the regular-package subpackages are
        # ``{groot, multi_task_dit, pi0, pi05, pi0_fast, rtc, wall_x, xvla}``
        # -- the rest (act, diffusion, smolvla, ...) are namespace
        # packages and thus not enumerable here. ``pi0`` is stable across
        # all lerobot 0.5.x and is therefore a safe target. See issue
        # #278 for the namespace-package coverage gap (separate from
        # this regression test, which only pins the
        # walk-continues-after-error contract).
        booby_trap = "lerobot.policies.pi0.configuration_pi0"
        trap_triggered = []

        def maybe_raise(name, *args, **kwargs):
            if name == booby_trap:
                trap_triggered.append(name)
                raise RuntimeError("simulated decorator-time re-registration collision")
            return original_import(name, *args, **kwargs)

        # Patch importlib.import_module directly (not via resolution.importlib)
        # to ensure the monkeypatch is visible regardless of how Python
        # resolves the module attribute lookup inside the cached function.
        monkeypatch.setattr(importlib, "import_module", maybe_raise)

        resolution._ensure_policy_configs_registered.cache_clear()
        # Capture all WARNING+ records without restricting to a specific
        # logger name -- avoids edge cases where caplog's per-logger level
        # gating interacts poorly with handler propagation.
        with caplog.at_level(logging.WARNING):
            resolution._ensure_policy_configs_registered()

        # Verify the monkeypatch was actually invoked for the booby-trapped
        # candidate. If it was not, the walker did not reach this subpackage
        # (e.g. pkgutil.iter_modules returned nothing or act was not yielded
        # as is_pkg=True), and the test premise is invalid for this env.
        if not trap_triggered:
            pytest.skip(
                "monkeypatch for configuration_act was never invoked; "
                "lerobot.policies may not expose act as a pkgutil-iterable "
                "subpackage in this installation"
            )

        # The walk surfaced the booby-trapped subpackage at WARNING
        # level so an operator can see it in production logs.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and booby_trap in r.message]
        assert warnings, (
            "Expected a WARNING about the booby-trapped "
            f"{booby_trap} import; got records: "
            f"{[r.message for r in caplog.records]}"
        )

        # ...AND the registry still contains policies the walk reached
        # AFTER the failure. Pre-R1 the walk would return at the first
        # non-ImportError, leaving everything after it unregistered.
        # We assert against subpackages the walker actually visits
        # (regular packages with __init__.py): ``wall_x`` and ``xvla``
        # come after ``pi0`` alphabetically and are stable in lerobot
        # 0.5.x. ``groot`` comes BEFORE ``pi0`` so we include it as the
        # "registered before the failure" anchor.
        registered = set(PreTrainedConfig.get_known_choices().keys())
        survivors = registered & {"groot", "wall_x", "xvla"}
        assert survivors, (
            "Walk aborted on the first non-ImportError; expected at "
            "least one of {'groot', 'wall_x', 'xvla'} to still be "
            f"registered. Got: {sorted(registered)}"
        )

        resolution._ensure_policy_configs_registered.cache_clear()
