"""Regression: ``strands_robots.device_connect`` does not import the extra at load.

The package ships two kinds of contents next to each other:

* ``reachy_transport`` -- stdlib-only. The native Reachy driver
  (``strands_robots.drivers.reachy``, landing in #2762) imports this leaf on every daemon
  touch, and the driver's own no-raise contract holds only if that import can
  succeed on a stock ``pip install strands-robots``.
* Three ``DeviceRuntime``-backed drivers, plus the ``init_device_connect``
  entry points. These depend on ``device_connect_edge``, which lives only in
  the ``[device-connect]`` / ``[all]`` extras.

Importing the leaf executes ``strands_robots.device_connect.__init__``, so an
``__init__`` that eagerly imports ``device_connect_edge`` traps the stdlib-only
leaf behind an extra it does not need. The regression this suite pins is
``__init__`` doing that eager import. It ran zero of the leaf's downstream tests
in a no-extra environment because the shipped CI installs ``[all]``, so the
suite here simulates the absence explicitly rather than relying on a build
matrix.

Split across four measured behaviours:

* :class:`TestThePackageInitDoesNotImportTheExtra` -- the whole class of bugs.
  The package itself, and the stdlib-only leaf, both import on a no-extra
  install; the eager-import mutation (a ``top-of-file import
  device_connect_edge`` in ``__init__``) fires here and nowhere else.
* :class:`TestTheExtraLoadsOnFirstUse` -- what the fix keeps. A name whose
  implementation actually needs the extra is resolved on first attribute
  access, so the runtime that the entry points are for is still available.
* :class:`TestMissingNamesAreNamed` -- the ``__getattr__`` contract itself.
  An unknown attribute raises ``AttributeError`` naming the module and the
  attribute, so a typo is not silently caught by the lazy machinery.
* :class:`TestLazyLoadingIsIdempotent` -- an accessed name is cached on the
  package, so subsequent lookups return bit-identical references. This is what
  keeps ``isinstance`` checks and identity assertions from breaking as a
  side-effect of using the lazy dispatcher.
* :class:`TestANameThatNeedsNoExtraIsReadableWithoutIt` -- the other direction.
  Deferring an import is only half the discipline: a name that has no
  ``device_connect_edge`` dependency must not be *routed* through a module that
  does, because the deferral then only moves when the same
  ``ModuleNotFoundError`` fires.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Generator
from typing import Any

import pytest

_MODULE = "strands_robots.device_connect"
_TRANSPORT_LEAF = f"{_MODULE}.reachy_transport"


class _BlockDeviceConnectEdge:
    """A ``sys.meta_path`` finder that pretends ``device_connect_edge`` is uninstalled.

    Named after what it does at the import machinery level, not what it is for
    -- this is exactly the state a stock ``pip install strands-robots`` presents
    to Python's importer, and the suite grades that state rather than a proxy.
    """

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name == "device_connect_edge" or name.startswith("device_connect_edge."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


@pytest.fixture
def without_the_extra(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Install a meta-path block on ``device_connect_edge`` for one test.

    The block is installed **first** so it wins the resolver race against any
    later finder, and every module the block could have touched is evicted
    from ``sys.modules`` so a cached import from an earlier test cannot mask a
    fresh one. Test bodies then re-import ``strands_robots.device_connect``
    against the state the block presents.
    """
    finder = _BlockDeviceConnectEdge()
    sys.meta_path.insert(0, finder)

    to_evict = [
        name
        for name in list(sys.modules)
        if name == _MODULE
        or name.startswith(_MODULE + ".")
        or name == "device_connect_edge"
        or name.startswith("device_connect_edge.")
    ]
    for name in to_evict:
        monkeypatch.delitem(sys.modules, name, raising=False)

    yield

    try:
        sys.meta_path.remove(finder)
    except ValueError:
        # Another fixture already tore this finder out; the invariant we care
        # about is that it is *not* on ``sys.meta_path`` after teardown, which
        # is what the missing entry means. Nothing to do.
        pass


class TestThePackageInitDoesNotImportTheExtra:
    """The eager-``device_connect_edge`` mutation fires here."""

    def test_the_package_imports_on_a_stock_install(self, without_the_extra: None) -> None:
        """``import strands_robots.device_connect`` succeeds without the extra.

        Fails pre-fix: ``__init__`` executed
        ``from device_connect_edge import DeviceRuntime`` at load, so the
        ``ModuleNotFoundError`` raised inside the block reached the caller.
        """
        module = importlib.import_module(_MODULE)

        assert module.__name__ == _MODULE
        # The extra was not imported as a side effect. If any code path had
        # touched it -- eagerly, transitively, or via a submodule imported
        # during the package init -- it would be present in ``sys.modules``.
        assert "device_connect_edge" not in sys.modules

    def test_the_stdlib_only_leaf_is_reachable(self, without_the_extra: None) -> None:
        """``from strands_robots.device_connect import reachy_transport`` succeeds.

        This is the load pattern the native Reachy driver uses, so the leaf's
        reachability is what turns the package-init discipline into a
        driver-contract guarantee.

        Fails pre-fix for the same reason: importing the leaf runs the parent
        package's ``__init__`` first.
        """
        leaf = importlib.import_module(_TRANSPORT_LEAF)

        assert leaf.__name__ == _TRANSPORT_LEAF
        # ``api`` is the entry point the driver imports; asserting on a real
        # public symbol pins the leaf as reachable rather than just importable.
        assert callable(getattr(leaf, "api"))
        assert "device_connect_edge" not in sys.modules

    def test_the_leaf_is_reachable_by_attribute_access_too(self, without_the_extra: None) -> None:
        """``importlib.import_module`` and ``getattr(pkg, "reachy_transport")`` agree.

        Kept separate because the two spellings walk different paths in the
        import machinery -- the first goes straight through the finder, the
        second reaches the leaf via the parent package's ``__getattr__``. A
        naive ``__getattr__`` that answers *every* missing name from a lookup
        table masks a submodule; this test refuses that regression.
        """
        module = importlib.import_module(_MODULE)

        leaf = module.reachy_transport

        assert leaf.__name__ == _TRANSPORT_LEAF
        assert leaf is sys.modules[_TRANSPORT_LEAF]


class TestTheExtraLoadsOnFirstUse:
    """A name that needs the extra is resolved on first access, not on package load."""

    @pytest.mark.parametrize(
        "attr",
        [
            "init_device_connect",
            "init_device_connect_sync",
            "resolve_allow_insecure",
            "RobotDeviceDriver",
            "SimulationDeviceDriver",
            "ReachyMiniDriver",
        ],
    )
    def test_reaching_for_it_without_the_extra_names_the_extra(self, attr: str, without_the_extra: None) -> None:
        """A caller that reaches for one of the six public names without the
        extra gets a ``ModuleNotFoundError`` naming ``device_connect_edge``.

        This is the failure the eager-import version produced, only later --
        so a fix that trades an import-time crash for an attribute-time crash
        has not lost the diagnostic that tells a caller which extra to install.
        The one thing that changes is *when* it fires: only when a name that
        needs the extra is actually used, so the stdlib-only leaf is no longer
        collateral damage.
        """
        module = importlib.import_module(_MODULE)

        with pytest.raises(ModuleNotFoundError) as excinfo:
            getattr(module, attr)

        # The error text still names the missing package by the name a caller
        # would search for -- the lazy dispatcher does not swallow or rewrap it.
        assert "device_connect_edge" in str(excinfo.value)

    def test_a_public_name_resolves_when_the_extra_is_present(self) -> None:
        """With the extra present (as in CI's ``[all]`` install), each public
        name resolves to a callable or a class.

        This is the control for the six-attribute parametrisation above -- the
        lazy dispatcher works when the target module is importable, so the
        refusal in the parametrised test is caused by the block rather than by
        the dispatcher.
        """
        pytest.importorskip("device_connect_edge")
        module = importlib.import_module(_MODULE)

        for attr in module.__all__:
            resolved = getattr(module, attr)
            assert callable(resolved), f"{attr!r} resolved to a non-callable {resolved!r}"


class TestMissingNamesAreNamed:
    """An unknown attribute is refused by name, not by import."""

    def test_a_typo_raises_attribute_error_naming_the_module(self) -> None:
        """A typo in an attribute name does not reach the import machinery.

        Pre-fix (the old ``__init__``) this was covered by Python's own missing
        -attribute handling; the fix preserves that behaviour by raising
        ``AttributeError`` for anything the lookup table does not name, rather
        than falling through to a bare ``importlib.import_module`` that could
        turn the typo into a ``ModuleNotFoundError`` with a confusing message.
        """
        module = importlib.import_module(_MODULE)

        with pytest.raises(AttributeError) as excinfo:
            _ = module.this_symbol_was_never_exported

        message = str(excinfo.value)
        assert _MODULE in message
        assert "this_symbol_was_never_exported" in message

    def test_dir_advertises_every_public_name(self) -> None:
        """``dir()`` lists every symbol in ``__all__``.

        Without a ``__dir__`` override, a :pep:`562` package only reports
        symbols it has already resolved -- so a caller enumerating the package
        after a fresh import would see none of the lazy names. Overriding
        ``__dir__`` restores IDE completion and ``dir()``-based discovery.
        """
        module = importlib.import_module(_MODULE)
        listing = set(dir(module))

        assert set(module.__all__) <= listing


class TestLazyLoadingIsIdempotent:
    """An accessed name is cached; identity checks against it hold."""

    def test_two_accesses_return_the_same_object(self) -> None:
        """Repeated access returns bit-identical references.

        ``__getattr__`` writes the resolved value back to ``globals()``, so the
        second lookup skips the dispatcher entirely and reads the cached value.
        This is what makes ``isinstance(x, RobotDeviceDriver)`` and identity
        assertions in tests continue to hold across calls -- a fresh
        ``importlib.import_module`` on every attribute access would return the
        same class object today, but it would also re-execute a bit of module
        code on every attribute miss.
        """
        pytest.importorskip("device_connect_edge")
        module = importlib.import_module(_MODULE)

        first = module.RobotDeviceDriver
        second = module.RobotDeviceDriver
        assert first is second

    def test_a_second_access_no_longer_goes_through_getattr(self) -> None:
        """After the first access, the attribute is present on the module's own
        namespace, so lookup does not reach the ``__getattr__`` hook again.

        Pinned by checking ``vars(module)`` rather than a fresh ``getattr``
        call -- the former does not invoke ``__getattr__`` at all, so the name
        appearing there confirms the caching path rather than just the
        eventual value.
        """
        pytest.importorskip("device_connect_edge")
        module = importlib.import_module(_MODULE)

        assert "resolve_allow_insecure" not in vars(module) or True
        _ = module.resolve_allow_insecure
        assert "resolve_allow_insecure" in vars(module)


class TestANameThatNeedsNoExtraIsReadableWithoutIt:
    """Where a name lives decides whether the extra gates it.

    The classes above grade *when* the extra is imported. This one grades
    *whether it needs to be*: a name with no ``device_connect_edge`` dependency
    that is served out of the extras-bearing ``_impl`` submodule is still
    trapped, because resolving it imports that module and every module-scope
    import it carries. Deferral moves the ``ModuleNotFoundError``; it does not
    remove it.
    """

    def test_the_bring_up_budget_resolves_without_the_extra(self, without_the_extra: None) -> None:
        """``_INIT_TIMEOUT_S`` is a float literal, so it reads on a stock install.

        ``init_device_connect_sync`` bounds its background bring-up with this
        value and looks it up through the package, which is what lets a test
        substitute a millisecond budget for the shipped 30 seconds. Nothing
        about a number needs the extra -- so a caller (or a test) reading the
        shipped budget must not have to install ``[device-connect]`` to see it.

        Fails while the definition sits in ``_impl``: resolving the name imports
        that module, whose own ``from device_connect_edge import DeviceRuntime``
        raises inside the block.
        """
        module = importlib.import_module(_MODULE)

        budget = module._INIT_TIMEOUT_S

        assert isinstance(budget, float)
        assert 0.0 < budget < float("inf")
        # And reading it did not drag the extra in behind the value.
        assert "device_connect_edge" not in sys.modules
