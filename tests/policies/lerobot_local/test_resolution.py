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
        """End-to-end registry completeness: after calling the helper,
        every lerobot 0.5.x built-in policy MUST be in the
        ``PreTrainedConfig`` choice registry.

        Note: this test does NOT install the stub first, so lerobot's
        eager ``policies/__init__.py`` may do some of the registration
        work via its own side-effect imports. The stub-active codepath
        (where the walker is the sole registration mechanism) is
        validated separately by
        ``test_namespace_package_policies_registered_after_stubbed_lerobot_policies``.
        This test pins the observable contract: regardless of how
        registration happens internally, the registry is complete.
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

    def test_namespace_package_policies_registered_after_stubbed_lerobot_policies(self):
        """Stub-active codepath must register subpackages laid out as PEP 420
        namespace packages (no ``__init__.py``).

        In lerobot 0.5.x, several subpackages of ``lerobot.policies`` are
        namespace packages: ``act/``, ``diffusion/``, ``smolvla/``,
        ``tdmpc/``, ``vqbet/``. ``pkgutil.iter_modules`` does not yield
        them with ``is_pkg=True``, so a walker that gates on
        ``is_pkg`` silently skips them on the stub-active codepath
        (the very codepath this helper exists to repair).
        Pre-fix this test fails with ``act`` (and friends) missing
        from the registry; post-fix the on-disk directory listing
        catches them and ``configuration_act`` is imported regardless
        of namespace-package layout. See issue #278 for the upstream
        layout context.
        """
        # ``act`` ships in every lerobot 0.5.x; ``importorskip`` only
        # skips the test if lerobot itself is missing (already gated
        # by the module-level ``importorskip("lerobot")``).
        pytest.importorskip("lerobot.policies")
        import sys

        snapshot = _snapshot_lerobot_modules()
        _purge_lerobot_modules(snapshot)
        try:
            from strands_robots.policies.lerobot_local.resolution import (
                _ensure_lerobot_policies_importable,
                _ensure_policy_configs_registered,
            )

            _ensure_lerobot_policies_importable()  # installs the stub
            _ensure_policy_configs_registered.cache_clear()
            _ensure_policy_configs_registered()

            from lerobot.configs.policies import PreTrainedConfig

            registered = set(PreTrainedConfig.get_known_choices().keys())
            # ``act`` is the canary that the previous canary-import
            # bootstrap also registered, so the regression test fails
            # loudly the moment the stub-active path drops it. The
            # other namespace-package subpackages live alongside it
            # in lerobot 0.5.x and SHOULD also land in the registry
            # post-fix (``expected_min`` only asserts ``act`` to keep
            # the test stable across lerobot minor versions; the
            # broader coverage is asserted by the non-stub
            # ``test_pkgutil_walk_registers_every_lerobot_policy_subpackage``).
            assert "act" in registered, (
                f"act missing after stub+walk; registered: {sorted(registered)}. "
                "Did the walker drop on-disk-directory enumeration of "
                "namespace-package subpackages?"
            )
        finally:
            _purge_lerobot_modules(_snapshot_lerobot_modules())
            sys.modules.update(snapshot)
            from strands_robots.policies.lerobot_local.resolution import (
                _ensure_policy_configs_registered,
            )

            _ensure_policy_configs_registered.cache_clear()

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

    def test_walk_continues_after_subpackage_decorator_failure(self, tmp_path, monkeypatch, caplog):
        """A subpackage whose ``configuration_*`` raises a non-ImportError
        (e.g. ``RuntimeError`` from a re-registration collision, or
        ``AttributeError`` from a renamed sibling attribute) MUST NOT
        abort the walk. Pre-R1 the helper caught only ``ImportError``,
        so a single buggy decorator on one subpackage would leave the
        registry permanently half-populated for the lifetime of the
        process (because ``@functools.cache`` then froze the failed
        state).

        This test constructs a synthetic ``lerobot.policies``-like
        namespace in a tmpdir with a booby-trapped subpackage that
        raises ``RuntimeError`` at import time, plus a healthy
        subpackage that should still register. This approach is immune
        to upstream lerobot layout changes (e.g. a subpackage
        transitioning from regular to namespace package) and never
        silently SKIPs.
        """
        import importlib
        import logging
        import sys
        import types

        from strands_robots.policies.lerobot_local import resolution

        # --- Build a synthetic lerobot.policies tree in tmpdir ---
        # Structure:
        #   tmp_path/
        #     healthy_policy/
        #       __init__.py           (empty, makes it a regular package)
        #       configuration_healthy_policy.py  (registers a fake config)
        #     broken_policy/
        #       __init__.py           (empty)
        #       configuration_broken_policy.py   (raises RuntimeError)
        #     also_healthy/
        #       __init__.py           (empty)
        #       configuration_also_healthy.py    (registers another fake)

        healthy_dir = tmp_path / "healthy_policy"
        healthy_dir.mkdir()
        (healthy_dir / "__init__.py").write_text("")
        (healthy_dir / "configuration_healthy_policy.py").write_text(
            "# Healthy configuration module -- import succeeds.\nREGISTERED = True\n"
        )

        broken_dir = tmp_path / "broken_policy"
        broken_dir.mkdir()
        (broken_dir / "__init__.py").write_text("")
        (broken_dir / "configuration_broken_policy.py").write_text(
            "raise RuntimeError('simulated decorator-time re-registration collision')\n"
        )

        also_healthy_dir = tmp_path / "also_healthy"
        also_healthy_dir.mkdir()
        (also_healthy_dir / "__init__.py").write_text("")
        (also_healthy_dir / "configuration_also_healthy.py").write_text(
            "# Another healthy configuration module.\nREGISTERED = True\n"
        )

        # We need 'lerobot' itself to remain so _ensure_lerobot_policies_importable
        # can find lerobot.__path__, but we replace lerobot.policies.
        fake_policies = types.ModuleType("lerobot.policies")
        fake_policies.__path__ = [str(tmp_path)]
        fake_policies.__package__ = "lerobot.policies"

        # Track which modules got imported through our synthetic tree
        imported_modules = []
        original_import = importlib.import_module

        def tracking_import(name, *args, **kwargs):
            if name.startswith("lerobot.policies."):
                # For our synthetic subpackages, manually handle the import
                parts = name.split(".")
                if len(parts) >= 3:
                    sub_name = parts[2]  # e.g. "healthy_policy"
                    sub_dir = tmp_path / sub_name
                    if sub_dir.is_dir():
                        if len(parts) == 3:
                            # Package import
                            mod = types.ModuleType(name)
                            mod.__path__ = [str(sub_dir)]
                            mod.__package__ = name
                            sys.modules[name] = mod
                            imported_modules.append(name)
                            return mod
                        elif len(parts) == 4:
                            # Submodule import (e.g. configuration_broken_policy)
                            module_name = parts[3]
                            module_file = sub_dir / f"{module_name}.py"
                            if module_file.exists():
                                source = module_file.read_text()
                                mod = types.ModuleType(name)
                                mod.__file__ = str(module_file)
                                mod.__package__ = ".".join(parts[:3])
                                # #280: record the ATTEMPT before exec. The
                                # broken_policy configuration raises at exec
                                # time; appending after exec would mean the
                                # assertion only matched the package-level
                                # fallback candidate, not the configuration
                                # module that actually triggered the trap.
                                imported_modules.append(name)
                                # Execute the source -- this is where broken_policy raises
                                exec(compile(source, str(module_file), "exec"), mod.__dict__)  # noqa: S102
                                sys.modules[name] = mod
                                return mod
                            raise ImportError(f"No module named '{name}'")
                # Fall through to real import for anything not in our tree
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", tracking_import)

        # Install our fake lerobot.policies. We must patch BOTH the
        # sys.modules entry AND the parent package attribute: the
        # production code does ``import lerobot.policies as _lr_policies``,
        # and ``import a.b as x`` binds ``x = getattr(a, "b")`` (the parent
        # attribute), NOT ``sys.modules["a.b"]``. If another test already
        # ran ``import lerobot.policies`` (e.g. test_embodiment_pipeline),
        # the real ``lerobot`` package retains a ``.policies`` attribute that
        # would otherwise shadow our sys.modules stub and make the walk see
        # the real tree (finding nothing new to import). Patching the parent
        # attribute closes that test-ordering leak.
        monkeypatch.setitem(sys.modules, "lerobot.policies", fake_policies)
        try:
            import lerobot as _real_lerobot

            monkeypatch.setattr(_real_lerobot, "policies", fake_policies, raising=False)
        except ImportError:
            pass  # lerobot not installed → import lerobot.policies uses sys.modules stub directly

        resolution._ensure_policy_configs_registered.cache_clear()

        with caplog.at_level(logging.WARNING):
            resolution._ensure_policy_configs_registered()

        # #280: the booby-trapped CONFIGURATION module specifically MUST
        # have been attempted -- not merely the package-level fallback.
        # Asserting on ``configuration_broken_policy`` uniquely pins the
        # R1-1 contract (non-ImportError in a configuration_* import does
        # not abort the walk) and stays correct even if the candidate
        # tuple is later reordered package-first.
        config_attempted = any(m.endswith("configuration_broken_policy") for m in imported_modules)
        assert config_attempted, f"The walker never attempted configuration_broken_policy; imported: {imported_modules}"

        # The walk surfaced the failure at WARNING level.
        warning_texts = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        trap_warnings = [t for t in warning_texts if "broken_policy" in t]
        assert trap_warnings, (
            f"Expected a WARNING about the booby-trapped broken_policy import; got warning messages: {warning_texts}"
        )

        # The healthy subpackages that come alphabetically before AND
        # after the broken one MUST have been imported -- proving the
        # walk continued past the failure.
        healthy_imported = any("healthy_policy" in m for m in imported_modules)
        also_healthy_imported = any("also_healthy" in m for m in imported_modules)
        assert healthy_imported and also_healthy_imported, (
            "Walk aborted on the first non-ImportError; expected both "
            "'healthy_policy' and 'also_healthy' to be attempted. "
            f"Imported: {imported_modules}"
        )

        resolution._ensure_policy_configs_registered.cache_clear()


def test_iter_modules_non_package_siblings_excluded(tmp_path):
    """Pin for R6-1: ``iter_modules`` non-package entries must NOT be walked.

    In lerobot 0.5.x, ``lerobot/policies/`` contains non-package siblings
    like ``factory.py``, ``utils.py``, ``pretrained.py``, ``pi_gemma.py``.
    If these are fed into the walker's candidate tuple, the package-level
    fallback (``lerobot.policies.factory``) succeeds and pulls in
    transformers/diffusers -- exactly the heavy import graph the stub
    mechanism exists to avoid.

    This test constructs a synthetic ``lerobot.policies``-like namespace
    with one regular-package subdir and one non-package ``.py`` file, runs
    the walker, and asserts only the package was walked.

    Pre-fix (without ``if _is_pkg:`` guard): the ``.py`` sibling would
    appear in the walker's candidates and the package-level fallback for
    it would be attempted.
    """
    import importlib
    import sys
    import types

    from strands_robots.policies.lerobot_local import resolution

    # Build a synthetic lerobot.policies-like directory:
    # tmp_path/
    #   real_policy/
    #     configuration_real_policy.py  -> registers the policy
    #   heavy_sibling.py  -> a non-package .py file that should NOT be imported
    real_pkg = tmp_path / "real_policy"
    real_pkg.mkdir()
    (real_pkg / "__init__.py").write_text("")
    (real_pkg / "configuration_real_policy.py").write_text("REGISTERED = True  # simulates decorator registration")

    # A non-package sibling (simulates factory.py / utils.py)
    (tmp_path / "heavy_sibling.py").write_text(
        "raise RuntimeError('heavy_sibling should never be imported by the walker')"
    )

    # Install a fake lerobot.policies module pointing at tmp_path
    fake_lr = types.ModuleType("lerobot")
    fake_lr.__path__ = []
    fake_lr_policies = types.ModuleType("lerobot.policies")
    fake_lr_policies.__path__ = [str(tmp_path)]
    fake_lr_policies.__name__ = "lerobot.policies"

    snapshot = _snapshot_lerobot_modules()
    _purge_lerobot_modules(snapshot)

    try:
        sys.modules["lerobot"] = fake_lr
        sys.modules["lerobot.policies"] = fake_lr_policies

        resolution._ensure_policy_configs_registered.cache_clear()

        # Track what gets imported
        original_import = importlib.import_module
        attempted_candidates = []

        def tracking_import(name, *args, **kwargs):
            attempted_candidates.append(name)
            return original_import(name, *args, **kwargs)

        import unittest.mock

        with unittest.mock.patch.object(importlib, "import_module", side_effect=tracking_import):
            resolution._ensure_policy_configs_registered()

        # The walker MUST have attempted configuration_real_policy (via
        # the directory-listing branch -- real_policy/ is a directory).
        assert any("real_policy" in c for c in attempted_candidates), (
            f"Expected 'real_policy' in walker candidates; got: {attempted_candidates}"
        )

        # The walker MUST NOT have attempted heavy_sibling (it's a .py
        # file, not a directory, and iter_modules should filter it with
        # is_pkg=True). If this assertion fails, the is_pkg guard is
        # missing and the non-package leak is back.
        assert not any("heavy_sibling" in c for c in attempted_candidates), (
            "Non-package sibling 'heavy_sibling' was walked by the resolver -- "
            "the is_pkg filter is missing. This would pull in transformers/diffusers "
            f"on production lerobot installs. Candidates attempted: {attempted_candidates}"
        )
    finally:
        # Purge any lerobot modules that were added during the test
        _purge_lerobot_modules(_snapshot_lerobot_modules())
        sys.modules.update(snapshot)
        resolution._ensure_policy_configs_registered.cache_clear()


def test_walk_continues_after_subpackage_decorator_failure_layout_independent(tmp_path, monkeypatch, caplog):
    """Layout-independent pin for the R1-1 walk-continues contract (#279).

    The companion ``test_walk_continues_after_subpackage_decorator_failure``
    exercises regular packages (with ``__init__.py``). This variant builds a
    PEP 420 *namespace-package* tree (no ``__init__.py`` in the subpackages)
    so the contract is pinned for the exact layout shape that motivated the
    directory-scan branch (``act``/``diffusion``/``smolvla`` in lerobot 0.5.x).

    A booby-trapped namespace subpackage whose ``configuration_*`` raises a
    non-ImportError MUST NOT abort the walk; a clean namespace subpackage that
    sorts after it MUST still be reached. No coupling to upstream lerobot.
    """
    import importlib
    import logging
    import sys
    import types

    from strands_robots.policies.lerobot_local import resolution

    # Namespace-package subpackages: NO __init__.py in either dir.
    (tmp_path / "trap").mkdir()
    (tmp_path / "trap" / "configuration_trap.py").write_text(
        "raise RuntimeError('simulated decorator-time re-registration collision')\n"
    )
    (tmp_path / "zclean").mkdir()  # sorts AFTER 'trap'
    (tmp_path / "zclean" / "configuration_zclean.py").write_text("REGISTERED = True\n")

    fake_policies = types.ModuleType("lerobot.policies")
    fake_policies.__path__ = [str(tmp_path)]
    fake_policies.__package__ = "lerobot.policies"
    fake_policies.__name__ = "lerobot.policies"

    attempted: list[str] = []
    original_import = importlib.import_module

    def tracking_import(name, *args, **kwargs):
        if name.startswith("lerobot.policies."):
            parts = name.split(".")
            if len(parts) == 4:
                sub_name, module_name = parts[2], parts[3]
                module_file = tmp_path / sub_name / f"{module_name}.py"
                if module_file.exists():
                    mod = types.ModuleType(name)
                    mod.__file__ = str(module_file)
                    mod.__package__ = ".".join(parts[:3])
                    attempted.append(name)  # record attempt before exec (#280 discipline)
                    exec(compile(module_file.read_text(), str(module_file), "exec"), mod.__dict__)  # noqa: S102
                    sys.modules[name] = mod
                    return mod
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    snapshot = _snapshot_lerobot_modules()
    _purge_lerobot_modules(snapshot)
    monkeypatch.setattr(importlib, "import_module", tracking_import)
    monkeypatch.setitem(sys.modules, "lerobot.policies", fake_policies)
    try:
        import lerobot as _real_lerobot

        monkeypatch.setattr(_real_lerobot, "policies", fake_policies, raising=False)
    except ImportError:
        # lerobot is an optional dependency; when it is not installed there is
        # no real package to patch and the fake module in sys.modules suffices.
        pass

    resolution._ensure_policy_configs_registered.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            resolution._ensure_policy_configs_registered()

        assert any(m.endswith("configuration_trap") for m in attempted), (
            f"walker never attempted the booby-trapped namespace config; attempted: {attempted}"
        )
        assert any(m.endswith("configuration_zclean") for m in attempted), (
            f"walk aborted on the trap; clean namespace subpackage never reached; attempted: {attempted}"
        )
        trap_warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "trap" in r.getMessage()
        ]
        assert trap_warnings, "expected a WARNING surfacing the booby-trapped namespace subpackage"
    finally:
        _purge_lerobot_modules(_snapshot_lerobot_modules())
        sys.modules.update(snapshot)
        resolution._ensure_policy_configs_registered.cache_clear()


def test_directory_scan_rejects_python_keyword_dirnames(tmp_path, monkeypatch):
    """#295: a subdirectory named after a Python keyword (``class``, ``for``,
    ``is``, ...) must be rejected by the directory-scan filter.

    ``str.isidentifier()`` returns True for keywords, but
    ``import lerobot.policies.class`` raises ``SyntaxError`` -- which is NOT an
    ``ImportError`` and would escape the per-candidate catch and abort the
    whole walk. The filter must mirror ``pkgutil`` and also reject keywords.

    Pre-fix (``if not name.isidentifier():`` only): ``class`` enters the
    candidate loop and the walker attempts to import it. Post-fix
    (``or keyword.iskeyword(name)``): ``class`` never reaches the loop.
    """
    import importlib
    import sys
    import types

    from strands_robots.policies.lerobot_local import resolution

    # A keyword-named dir with a configuration module, plus a valid one.
    (tmp_path / "class").mkdir()
    (tmp_path / "class" / "configuration_class.py").write_text("REGISTERED = True\n")
    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "configuration_valid.py").write_text("REGISTERED = True\n")

    fake_policies = types.ModuleType("lerobot.policies")
    fake_policies.__path__ = [str(tmp_path)]
    fake_policies.__package__ = "lerobot.policies"
    fake_policies.__name__ = "lerobot.policies"

    attempted: list[str] = []
    original_import = importlib.import_module

    def tracking_import(name, *args, **kwargs):
        attempted.append(name)
        if name.startswith("lerobot.policies."):
            parts = name.split(".")
            if len(parts) == 4:
                module_file = tmp_path / parts[2] / f"{parts[3]}.py"
                if module_file.exists():
                    mod = types.ModuleType(name)
                    mod.__file__ = str(module_file)
                    exec(compile(module_file.read_text(), str(module_file), "exec"), mod.__dict__)  # noqa: S102
                    sys.modules[name] = mod
                    return mod
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    snapshot = _snapshot_lerobot_modules()
    _purge_lerobot_modules(snapshot)
    monkeypatch.setattr(importlib, "import_module", tracking_import)
    monkeypatch.setitem(sys.modules, "lerobot.policies", fake_policies)
    try:
        import lerobot as _real_lerobot

        monkeypatch.setattr(_real_lerobot, "policies", fake_policies, raising=False)
    except ImportError:
        # lerobot is an optional dependency; when it is not installed there is
        # no real package to patch and the fake module in sys.modules suffices.
        pass

    resolution._ensure_policy_configs_registered.cache_clear()
    try:
        # Must not raise (pre-fix, the keyword dir is walked; depending on the
        # import machinery the bare ``import lerobot.policies.class`` raises
        # SyntaxError and aborts). Post-fix the keyword dir is filtered out.
        resolution._ensure_policy_configs_registered()

        assert not any("policies.class" in c for c in attempted), (
            "keyword-named dir 'class' reached the candidate loop; the "
            f"keyword.iskeyword filter is missing. Attempted: {attempted}"
        )
        assert any("policies.valid" in c for c in attempted), (
            f"valid subpackage should still be walked; attempted: {attempted}"
        )
    finally:
        _purge_lerobot_modules(_snapshot_lerobot_modules())
        sys.modules.update(snapshot)
        resolution._ensure_policy_configs_registered.cache_clear()


class TestResolvePolicyClassFromHub:
    """Behavioral tests for the public ``resolve_policy_class_from_hub`` entry.

    This is the function ``LerobotLocalPolicy`` calls to turn a HuggingFace
    repo id into a ``(PolicyClass, policy_type)`` pair. It has two strategies:
    a draccus ``PreTrainedConfig.from_pretrained`` path (preferred), and a
    manual ``config.json`` fallback for third-party policies that draccus
    cannot decode. The tests below assert the observable contract of each
    branch -- what the function returns or raises -- not its internals.
    """

    def test_draccus_path_returns_class_and_type(self, monkeypatch):
        """Strategy 1: a config whose ``type`` decodes via draccus yields the
        matching policy class and the type string read off the config."""
        from strands_robots.policies.lerobot_local import resolution

        class _FakeConfig:
            type = "act"

        class _FakeACTPolicy:
            pass

        def _from_pretrained(_path):
            return _FakeConfig()

        monkeypatch.setattr(
            "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
            staticmethod(_from_pretrained),
        )
        monkeypatch.setattr(resolution, "_ensure_policy_configs_registered", lambda: None)
        monkeypatch.setattr(
            resolution,
            "resolve_policy_class_by_name",
            lambda policy_type: _FakeACTPolicy if policy_type == "act" else None,
        )

        policy_class, policy_type = resolution.resolve_policy_class_from_hub("some/act-repo")

        assert policy_class is _FakeACTPolicy
        assert policy_type == "act"

    def test_draccus_failure_falls_back_to_manual_config(self, monkeypatch):
        """Strategy 2: when draccus raises a decode-style error, the function
        falls through to reading ``config.json`` for the ``type`` field."""
        from strands_robots.policies.lerobot_local import resolution

        class _FakeCustomPolicy:
            pass

        def _boom(_path):
            raise ValueError("draccus cannot decode this third-party config")

        monkeypatch.setattr(
            "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
            staticmethod(_boom),
        )
        monkeypatch.setattr(resolution, "_ensure_policy_configs_registered", lambda: None)
        monkeypatch.setattr(resolution, "_read_policy_type_from_config", lambda _p: "custom_type")
        monkeypatch.setattr(
            resolution,
            "resolve_policy_class_by_name",
            lambda policy_type: _FakeCustomPolicy if policy_type == "custom_type" else None,
        )

        policy_class, policy_type = resolution.resolve_policy_class_from_hub("third/party-repo")

        assert policy_class is _FakeCustomPolicy
        assert policy_type == "custom_type"

    def test_manual_fallback_without_type_raises_value_error(self, monkeypatch):
        """When both draccus and ``config.json`` fail to yield a type, the
        function raises ``ValueError`` telling the caller to pass it explicitly."""
        from strands_robots.policies.lerobot_local import resolution

        def _boom(_path):
            raise ValueError("undecodable")

        monkeypatch.setattr(
            "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
            staticmethod(_boom),
        )
        monkeypatch.setattr(resolution, "_ensure_policy_configs_registered", lambda: None)
        monkeypatch.setattr(resolution, "_read_policy_type_from_config", lambda _p: None)

        with pytest.raises(ValueError, match="Could not determine policy type"):
            resolution.resolve_policy_class_from_hub("mystery/repo")

    def test_missing_lerobot_import_error_propagates(self, monkeypatch):
        """An ``ImportError`` (lerobot absent) must NOT be swallowed -- it is a
        real, terminal error, distinct from a config that simply fails to decode."""
        import builtins

        from strands_robots.policies.lerobot_local import resolution

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "lerobot.configs.policies":
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError, match="no lerobot"):
            resolution.resolve_policy_class_from_hub("any/repo")

    def test_non_draccus_unexpected_exception_propagates(self, monkeypatch):
        """A non-draccus, non-listed exception from ``from_pretrained`` must
        propagate rather than being silently routed to the manual fallback."""
        from strands_robots.policies.lerobot_local import resolution

        class _WeirdError(Exception):
            pass

        def _boom(_path):
            raise _WeirdError("not a draccus error and not in the handled tuple")

        monkeypatch.setattr(
            "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
            staticmethod(_boom),
        )
        monkeypatch.setattr(resolution, "_ensure_policy_configs_registered", lambda: None)

        with pytest.raises(_WeirdError):
            resolution.resolve_policy_class_from_hub("weird/repo")


class TestReadPolicyTypeFromConfig:
    """Behavioral tests for ``_read_policy_type_from_config`` -- the manual
    ``config.json`` reader used as the third-party fallback. It checks a local
    directory first, then attempts a HuggingFace Hub download."""

    def test_reads_type_from_local_config_json(self, tmp_path):
        """A local directory containing ``config.json`` with a ``type`` field
        returns that type without touching the network."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(json.dumps({"type": "diffusion"}))

        assert resolution._read_policy_type_from_config(str(tmp_path)) == "diffusion"

    def test_local_config_without_type_returns_none(self, tmp_path):
        """A local ``config.json`` lacking a ``type`` field returns ``None``."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(json.dumps({"foo": "bar"}))

        assert resolution._read_policy_type_from_config(str(tmp_path)) is None

    def test_hub_download_failure_returns_none(self, monkeypatch):
        """When the path is not a local dir and the Hub download raises an
        OSError, the reader logs a warning and returns ``None`` (no crash)."""
        from strands_robots.policies.lerobot_local import resolution

        def _boom(*_args, **_kwargs):
            raise OSError("offline / repo not found")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

        assert resolution._read_policy_type_from_config("nonexistent/repo-id") is None

    def test_model_type_known_override_resolves(self, tmp_path):
        """A config with no ``type`` but a known ``model_type`` (``molmoact2``)
        resolves through ``_KNOWN_MODEL_TYPE_MAP`` rather than raising."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))

        assert resolution._read_policy_type_from_config(str(tmp_path)) == "molmoact2"

    def test_model_type_unknown_used_verbatim(self, tmp_path):
        """An unknown ``model_type`` (no override) is returned as-is so callers
        can still attempt class resolution by that name."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "act"}))

        assert resolution._read_policy_type_from_config(str(tmp_path)) == "act"

    def test_auto_map_class_name_resolves(self, tmp_path):
        """When only an ``auto_map`` is present, a recognized modeling class
        (``MolmoAct2ForConditionalGeneration``) maps to its policy type."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "auto_map": {
                        "AutoConfig": "configuration_molmoact2.MolmoAct2Config",
                        "AutoModelForImageTextToText": ("modeling_molmoact2.MolmoAct2ForConditionalGeneration"),
                    }
                }
            )
        )

        assert resolution._read_policy_type_from_config(str(tmp_path)) == "molmoact2"

    def test_type_field_wins_over_model_type(self, tmp_path):
        """The canonical lerobot ``type`` field takes precedence over
        ``model_type`` when both are present."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(json.dumps({"type": "diffusion", "model_type": "molmoact2"}))

        assert resolution._read_policy_type_from_config(str(tmp_path)) == "diffusion"

    def test_no_type_model_type_or_known_auto_map_returns_none(self, tmp_path):
        """A config with none of ``type``/``model_type`` and only unrecognized
        ``auto_map`` entries returns ``None`` (caller raises a clear error)."""
        import json

        from strands_robots.policies.lerobot_local import resolution

        (tmp_path / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModel": "modeling_unknownthing.UnknownThingModel"}})
        )

        assert resolution._read_policy_type_from_config(str(tmp_path)) is None

    @pytest.mark.slow
    def test_real_molmoact2_repo_resolves_model_type(self):
        """End-to-end against the live HF repo: ``allenai/MolmoAct2-SO100_101``
        sets ``model_type: molmoact2`` with ``type`` unset, and must resolve.

        Skipped offline (network failure surfaces as ``None`` from the reader's
        ``OSError`` guard, which would make this a flaky false-negative)."""
        from strands_robots.policies.lerobot_local import resolution

        result = resolution._read_policy_type_from_config("allenai/MolmoAct2-SO100_101")
        if result is None:
            pytest.skip("HF Hub unreachable; cannot exercise live repo")
        assert result == "molmoact2"


class TestResolvePolicyClassByNameFallbackLadder:
    """Behavioral tests for the resolution ladder in
    ``resolve_policy_class_by_name`` AFTER the LeRobot 0.5+ ``modeling_<type>``
    convention (Strategy 1) misses.

    Real LeRobot installs differ by version: some expose a concrete policy
    class at the package level, older ones only ship the legacy
    ``lerobot.policies.factory.get_policy_class`` factory, and the abstract
    ``PreTrainedPolicy`` is the last resort. These fall-through strategies are
    exactly what shields callers from LeRobot layout drift, so each rung -- and
    the exhaustion error -- is pinned here. All run without importing real
    lerobot: the package chain is faked in ``sys.modules`` and Strategy 1/2's
    ``importlib.import_module`` is stubbed to miss.
    """

    @pytest.fixture(autouse=True)
    def _isolate_resolution(self, monkeypatch):
        """Neutralize the stub installer and inject a minimal fake ``lerobot``
        package chain so the ladder's ``from lerobot.policies...`` imports
        resolve from ``sys.modules`` alone (no heavy real __init__ executes)."""
        import sys
        import types

        from strands_robots.policies.lerobot_local import resolution

        monkeypatch.setattr(resolution, "_ensure_lerobot_policies_importable", lambda: None)

        lerobot_mod = types.ModuleType("lerobot")
        lerobot_mod.__path__ = []  # mark as package
        policies_mod = types.ModuleType("lerobot.policies")
        policies_mod.__path__ = []
        monkeypatch.setitem(sys.modules, "lerobot", lerobot_mod)
        monkeypatch.setitem(sys.modules, "lerobot.policies", policies_mod)
        return resolution

    def test_strategy2_package_level_import_returns_policy_class(self, _isolate_resolution, monkeypatch):
        """When ``modeling_<type>`` is absent but the package itself re-exports
        a ``*Policy`` class with ``from_pretrained`` (an alternate LeRobot
        layout), Strategy 2's package-level import returns that class."""
        import types

        resolution = _isolate_resolution

        class FakeACTPolicy:
            @classmethod
            def from_pretrained(cls):
                return cls()

        pkg = types.ModuleType("lerobot.policies.act")
        pkg.ACTPolicy = FakeACTPolicy

        def fake_import(name):
            if name == "lerobot.policies.act":
                return pkg
            raise ImportError(name)  # Strategy 1 modeling_* misses

        monkeypatch.setattr(resolution.importlib, "import_module", fake_import)

        assert resolution.resolve_policy_class_by_name("act") is FakeACTPolicy

    def test_strategy2_skips_pretrained_policy_and_attrs_without_from_pretrained(
        self, _isolate_resolution, monkeypatch
    ):
        """Strategy 2 must ignore the abstract ``PreTrainedPolicy`` base and any
        ``*Policy``-named attribute lacking ``from_pretrained``, returning only a
        genuine concrete checkpoint-loadable class."""
        import types

        resolution = _isolate_resolution

        class PreTrainedPolicy:  # base class -- must be skipped by name guard
            @classmethod
            def from_pretrained(cls):
                return cls()

        class NotARealPolicy:  # ends with "Policy" but cannot load checkpoints
            pass

        class DiffusionPolicy:
            @classmethod
            def from_pretrained(cls):
                return cls()

        pkg = types.ModuleType("lerobot.policies.diffusion")
        pkg.PreTrainedPolicy = PreTrainedPolicy
        pkg.NotARealPolicy = NotARealPolicy
        pkg.DiffusionPolicy = DiffusionPolicy

        def fake_import(name):
            if name == "lerobot.policies.diffusion":
                return pkg
            raise ImportError(name)

        monkeypatch.setattr(resolution.importlib, "import_module", fake_import)

        assert resolution.resolve_policy_class_by_name("diffusion") is DiffusionPolicy

    def test_strategy3_legacy_factory_returns_policy_class(self, _isolate_resolution, monkeypatch):
        """On LeRobot <0.4 layouts where neither the ``modeling_*`` submodule nor
        a package-level class exists, Strategy 3 delegates to the legacy
        ``lerobot.policies.factory.get_policy_class`` factory."""
        import sys
        import types

        resolution = _isolate_resolution

        class LegacyTDMPCPolicy:
            pass

        def fake_import(name):
            raise ImportError(name)  # Strategy 1 + 2 both miss

        monkeypatch.setattr(resolution.importlib, "import_module", fake_import)

        factory_mod = types.ModuleType("lerobot.policies.factory")
        factory_mod.get_policy_class = lambda policy_type: LegacyTDMPCPolicy
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_mod)

        assert resolution.resolve_policy_class_by_name("tdmpc") is LegacyTDMPCPolicy

    def test_strategy4_concrete_pretrained_policy_is_last_resort(self, _isolate_resolution, monkeypatch):
        """When every type-specific path misses and the legacy factory is gone,
        Strategy 4 returns ``PreTrainedPolicy`` itself -- but only because it is
        concrete (not abstract) in this fake layout."""
        import sys
        import types

        resolution = _isolate_resolution

        def fake_import(name):
            raise ImportError(name)

        monkeypatch.setattr(resolution.importlib, "import_module", fake_import)

        # Strategy 3: factory module present but without get_policy_class ->
        # the ``from ... import get_policy_class`` raises ImportError (caught).
        factory_mod = types.ModuleType("lerobot.policies.factory")
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_mod)

        class PreTrainedPolicy:  # concrete in this fake -> usable fallback
            pass

        pretrained_mod = types.ModuleType("lerobot.policies.pretrained")
        pretrained_mod.PreTrainedPolicy = PreTrainedPolicy
        monkeypatch.setitem(sys.modules, "lerobot.policies.pretrained", pretrained_mod)

        assert resolution.resolve_policy_class_by_name("mystery") is PreTrainedPolicy

    def test_all_strategies_exhausted_raises_importerror_with_guidance(self, _isolate_resolution, monkeypatch):
        """When no rung resolves a class -- including an abstract
        ``PreTrainedPolicy`` that Strategy 4 must reject -- the function raises
        ``ImportError`` naming the type and the strategies it tried, rather than
        returning the abstract base or a silent ``None``."""
        import abc
        import sys
        import types

        resolution = _isolate_resolution

        def fake_import(name):
            raise ImportError(name)

        monkeypatch.setattr(resolution.importlib, "import_module", fake_import)

        factory_mod = types.ModuleType("lerobot.policies.factory")  # no get_policy_class
        monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_mod)

        class AbstractPreTrainedPolicy(abc.ABC):
            @abc.abstractmethod
            def forward(self): ...

        pretrained_mod = types.ModuleType("lerobot.policies.pretrained")
        pretrained_mod.PreTrainedPolicy = AbstractPreTrainedPolicy
        monkeypatch.setitem(sys.modules, "lerobot.policies.pretrained", pretrained_mod)

        with pytest.raises(ImportError, match="ghost-policy"):
            resolution.resolve_policy_class_by_name("ghost-policy")
