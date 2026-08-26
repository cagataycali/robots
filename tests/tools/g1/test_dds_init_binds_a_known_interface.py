"""ensure_dds records the interface it bound, not the one it was asked for.

The G1 DDS helper's first stated requirement is that
``ChannelFactoryInitialize`` runs "exactly once per process, with a known
network interface", and it keeps a module-level record of that interface so a
later caller asking for a different one is refused rather than silently
re-bound onto the wrong NIC.

The record is only worth refusing on if it is true. ``ChannelFactory.Init``
short-circuits on ``if __initialized: return True`` and never looks at its
``networkInterface`` argument, so a second ``ChannelFactoryInitialize`` returns
normally without binding anything - a no-op that is indistinguishable, at the
call site, from a successful bind. A process where anything else brought the
bus up first therefore had ``ensure_dds`` report success and record an
interface the bus was never on, and a later refusal then quoted that
fabricated interface back at the caller.

Every cell here drives a fake ``unitree_sdk2py.core.channel``: binding the
real factory is a process-wide side effect with no ``Shutdown``, so a suite
that did it once would change what every later test observes.
:class:`TestTheRealSdkBehavesTheWayTheDoubleDoes` pins the double against the
installed SDK so it cannot drift into agreeing with a bug, and skips when the
SDK is absent (Thor, CI) rather than passing vacuously.
"""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.tools.g1 import _g1_common, reset_dds_state

#: The attribute name the SDK mangles ``ChannelFactory.__initialized`` to.
#: Stated here rather than read from the module so a cell that asserts the
#: module agrees with the SDK is comparing two independent statements.
SDK_BOUND_ATTR = "_ChannelFactory__initialized"


class _FakeFactory:
    """Stands in for ``ChannelFactory``: one class-level "bus is up" flag."""


def _fake_channel_module(*, already_bound: bool, bound_attr: str = SDK_BOUND_ATTR) -> Any:
    """A fake ``unitree_sdk2py.core.channel`` that records init calls.

    Mirrors the one behaviour that matters: ``ChannelFactoryInitialize``
    returns ``None`` whether or not it bound anything, so the caller cannot
    tell a bind from a no-op by looking at the call.

    Args:
        already_bound: Whether the factory reports itself already bound.
        bound_attr: The attribute the flag is published on. A build that
            publishes it elsewhere is what ``None`` from the probe means.

    Returns:
        The fake module; its ``init_calls`` list records every call.
    """
    module = types.ModuleType("unitree_sdk2py.core.channel")
    factory = type("ChannelFactory", (_FakeFactory,), {bound_attr: already_bound})
    calls: list[tuple[int, str]] = []

    def channel_factory_initialize(domain_id: int, network_interface: str) -> None:
        calls.append((domain_id, network_interface))
        setattr(factory, bound_attr, True)

    module.ChannelFactory = factory  # type: ignore[attr-defined]
    module.ChannelFactoryInitialize = channel_factory_initialize  # type: ignore[attr-defined]
    module.init_calls = calls  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Install a fake SDK channel module and clear the recorded DDS state.

    Yields a callable that installs a fake with a chosen initial flag and
    returns it, so a cell picks "the bus was already up" or "the bus is down"
    without reaching into ``sys.modules`` itself.
    """
    reset_dds_state()

    def install(*, already_bound: bool, bound_attr: str = SDK_BOUND_ATTR) -> Any:
        module = _fake_channel_module(already_bound=already_bound, bound_attr=bound_attr)
        parent = types.ModuleType("unitree_sdk2py.core")
        parent.channel = module  # type: ignore[attr-defined]
        grandparent = types.ModuleType("unitree_sdk2py")
        grandparent.core = parent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unitree_sdk2py", grandparent)
        monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", parent)
        monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", module)
        return module

    yield install
    reset_dds_state()


# =========================================================================
# The premise: a no-op and a bind look the same at the call site.          #
# =========================================================================


class TestTheDoubleIsFaithfulToTheCallSite:
    """The fake reproduces the property that makes the defect possible."""

    def test_a_second_initialize_returns_normally_without_binding(self, fake_sdk: Any) -> None:
        """A fake already-bound factory still accepts an init call quietly."""
        module = fake_sdk(already_bound=True)
        assert module.ChannelFactoryInitialize(0, "wlan0") is None

    def test_the_probe_reads_the_flag_the_factory_publishes(self, fake_sdk: Any) -> None:
        """Both flag states are reported, so the probe is not a constant."""
        assert _g1_common._sdk_factory_already_bound(fake_sdk(already_bound=True)) is True
        assert _g1_common._sdk_factory_already_bound(fake_sdk(already_bound=False)) is False


# =========================================================================
# The regression: a bus this process did not bind is not recorded as ours. #
# =========================================================================


class TestABusBoundElsewhereIsNotRecordedAsOurs:
    """``ensure_dds`` refuses rather than claiming an interface it cannot confirm."""

    def test_a_factory_bound_elsewhere_is_refused(self, fake_sdk: Any) -> None:
        """A bind that cannot be confirmed is a reason, not a success."""
        fake_sdk(already_bound=True)

        assert _g1_common.ensure_dds("eth-not-the-bound-one") is not None

    def test_the_refusal_names_the_cause(self, fake_sdk: Any) -> None:
        """ "the bus is down" and "someone else brought it up" need different fixes."""
        fake_sdk(already_bound=True)

        err = _g1_common.ensure_dds("eth-not-the-bound-one")

        assert err is not None
        assert "already initialised outside ensure_dds" in err

    def test_the_refusal_names_the_interface_it_could_not_confirm(self, fake_sdk: Any) -> None:
        """A caller with two interfaces needs to know which one was refused."""
        fake_sdk(already_bound=True)

        err = _g1_common.ensure_dds("eth-not-the-bound-one")

        assert err is not None
        assert "eth-not-the-bound-one" in err

    def test_the_refusal_names_the_call_to_remove(self, fake_sdk: Any) -> None:
        """The remedy is actionable: the caller is told what to drop."""
        fake_sdk(already_bound=True)

        err = _g1_common.ensure_dds("eth-not-the-bound-one")

        assert err is not None
        assert "ChannelFactoryInitialize" in err

    def test_an_interface_that_was_not_bound_is_never_recorded(self, fake_sdk: Any) -> None:
        """The record stays empty, so no later refusal can quote a false NIC."""
        fake_sdk(already_bound=True)

        _g1_common.ensure_dds("eth-not-the-bound-one")

        assert _g1_common._dds_state["initialized"] is False
        assert _g1_common._dds_state["interface"] is None

    def test_no_second_initialize_is_issued_against_a_bound_factory(self, fake_sdk: Any) -> None:
        """The probe runs before the call, so the no-op is never made."""
        module = fake_sdk(already_bound=True)

        _g1_common.ensure_dds("eth-not-the-bound-one")

        assert module.init_calls == []

    def test_a_build_that_refuses_a_second_init_is_refused_too(self, fake_sdk: Any) -> None:
        """An SDK that raises "already initialized" is the same situation.

        Such a build reports the bus is up on an interface this process did
        not choose, so recording the requested one would fabricate the same
        fact the probe above refuses to fabricate.
        """
        module = fake_sdk(already_bound=False)

        def refuse(domain_id: int, network_interface: str) -> None:
            raise Exception("factory already initialized")

        module.ChannelFactoryInitialize = refuse

        err = _g1_common.ensure_dds("wlan0")

        assert err is not None
        assert "already" in err
        assert "wlan0" in err
        assert _g1_common._dds_state["interface"] is None


# =========================================================================
# What is unchanged: a bind this process performed still reads as ours.    #
# =========================================================================


class TestAnAttestedBindIsUnchanged:
    """Every cell here holds on the pre-fix code as well; that is the point."""

    def test_a_first_bind_records_the_interface_it_bound(self, fake_sdk: Any) -> None:
        """Success is reported and the interface is recorded."""
        module = fake_sdk(already_bound=False)

        assert _g1_common.ensure_dds("eth0") is None
        assert module.init_calls == [(0, "eth0")]
        assert _g1_common._dds_state["initialized"] is True
        assert _g1_common._dds_state["interface"] == "eth0"

    def test_a_repeat_call_with_the_same_interface_is_idempotent(self, fake_sdk: Any) -> None:
        """The second call succeeds without a second init."""
        module = fake_sdk(already_bound=False)
        assert _g1_common.ensure_dds("eth0") is None

        assert _g1_common.ensure_dds("eth0") is None
        assert module.init_calls == [(0, "eth0")]

    def test_a_repeat_call_with_another_interface_is_refused(self, fake_sdk: Any) -> None:
        """The refusal names both the bound interface and the requested one."""
        fake_sdk(already_bound=False)
        assert _g1_common.ensure_dds("eth0") is None

        err = _g1_common.ensure_dds("wlan0")

        assert err is not None
        assert "eth0" in err
        assert "wlan0" in err
        assert _g1_common._dds_state["interface"] == "eth0"

    def test_a_failing_bind_is_reported_and_records_nothing(self, fake_sdk: Any) -> None:
        """A genuine init failure keeps its own wording."""
        module = fake_sdk(already_bound=False)

        def fail(domain_id: int, network_interface: str) -> None:
            raise Exception("channel factory init error.")

        module.ChannelFactoryInitialize = fail

        err = _g1_common.ensure_dds("eth-nonexistent")

        assert err is not None
        assert "ChannelFactoryInitialize failed" in err
        assert _g1_common._dds_state["initialized"] is False


class TestAnUnreadableSdkKeepsTheOlderBehaviour:
    """A build that does not publish the flag must not be refused blindly."""

    def test_the_probe_reports_unknown_rather_than_guessing(self, fake_sdk: Any) -> None:
        """No flag where we look means "cannot tell", not "already bound"."""
        module = fake_sdk(already_bound=True, bound_attr="_SomeOtherName__initialized")

        assert _g1_common._sdk_factory_already_bound(module) is None

    def test_a_bind_still_succeeds_when_the_flag_cannot_be_read(self, fake_sdk: Any) -> None:
        """The caller keeps the behaviour it had before the probe existed."""
        module = fake_sdk(already_bound=True, bound_attr="_SomeOtherName__initialized")

        assert _g1_common.ensure_dds("eth0") is None
        assert module.init_calls == [(0, "eth0")]
        assert _g1_common._dds_state["interface"] == "eth0"

    def test_a_module_without_a_factory_reports_unknown(self) -> None:
        """A channel module with no ``ChannelFactory`` at all is not a crash."""
        assert _g1_common._sdk_factory_already_bound(types.ModuleType("bare")) is None


# =========================================================================
# The double against the installed SDK - skips without it, never passes    #
# vacuously.                                                              #
# =========================================================================


class TestTheRealSdkBehavesTheWayTheDoubleDoes:
    """The three SDK facts the fix rests on, read off the installed package."""

    @staticmethod
    def _channel_module() -> Any:
        return pytest.importorskip(
            "unitree_sdk2py.core.channel",
            reason="unitree_sdk2py is absent here; the double stands in for it",
        )

    def test_the_factory_publishes_its_bound_flag_where_the_probe_looks(self) -> None:
        """A rename in the SDK fails here rather than silently disabling the probe."""
        channel = self._channel_module()

        flag = getattr(channel.ChannelFactory, SDK_BOUND_ATTR, None)

        assert isinstance(flag, bool), f"{SDK_BOUND_ATTR} is not a bool on this SDK build"
        assert _g1_common._SDK_FACTORY_BOUND_ATTR == SDK_BOUND_ATTR

    def test_init_short_circuits_without_looking_at_the_interface(self) -> None:
        """``Init`` returns early on the flag, so a re-init binds nothing."""
        channel = self._channel_module()

        source = inspect.getsource(channel.ChannelFactory.Init)
        # Everything before the lock is the short-circuit; drop the signature
        # line so the interface parameter's own declaration is not counted.
        short_circuit = "\n".join(source.split("with ", 1)[0].splitlines()[1:])

        assert "__initialized" in short_circuit
        assert "return True" in short_circuit
        assert "networkInterface" not in short_circuit

    def test_initialize_discards_the_bool_init_returns(self) -> None:
        """Nothing public distinguishes a bind from a no-op.

        ``ChannelFactoryInitialize`` raises on a falsy ``Init`` and otherwise
        returns ``None``, so the truthy branch carries no information - which
        is why the flag has to be read directly.
        """
        channel = self._channel_module()

        source = inspect.getsource(channel.ChannelFactoryInitialize)

        assert "raise" in source
        assert "return" not in source.split("raise", 1)[1]
