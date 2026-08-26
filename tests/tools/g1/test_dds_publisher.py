"""DDSPublisher: the write half of the G1 DDS engine.

Every SDK touch is mocked. The subscriber set's tests already pin the
read path against the shared lock and lazy-import invariants; this suite
pins the write path against the *same* invariants, so a future change that
loosens either half of the engine fails here and there together.

No G1 hardware, no ``unitree_sdk2py`` on this box - the SDK's ``ChannelPublisher``
is injected into the module namespace under a monkeypatch, then the real
lazy import is restored. That is the shape the module invites: a module-level
``import unitree_sdk2py`` would fail on Thor and CI, so the tests must exercise
the lazy path rather than the cached one.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.tools.g1 import _dds_engine, reset_dds_state
from strands_robots.tools.g1._dds_engine import DDSPublisher

# =========================================================================
# Fixtures - a fake unitree_sdk2py that records what a caller did.        #
# =========================================================================


class _FakeChannelPublisher:
    """Records init/write calls; the driver's contract with the SDK.

    Recreated per fixture so tests do not share state, but every instance
    exposes the same three fields the driver actually reads: constructor
    args, whether Init was called, and the messages that were written.
    """

    def __init__(self, topic: str, message_class: type) -> None:
        self.topic = topic
        self.message_class = message_class
        self.init_calls = 0
        self.writes: list[Any] = []
        self.write_should_raise: Exception | None = None

    def Init(self) -> None:
        self.init_calls += 1

    def Write(self, message: Any) -> None:
        if self.write_should_raise is not None:
            raise self.write_should_raise
        self.writes.append(message)


class _RecordingChannelFactory:
    """Records ``ChannelPublisher`` constructions across a test.

    Instances built during a test are appended to :attr:`built`, so a test
    asserting exactly-once construction reads ``len(built)`` rather than
    poking private state on the publisher set.
    """

    def __init__(self, raise_on_construct: Exception | None = None) -> None:
        self.built: list[_FakeChannelPublisher] = []
        self.raise_on_construct = raise_on_construct

    def __call__(self, topic: str, message_class: type) -> _FakeChannelPublisher:
        if self.raise_on_construct is not None:
            raise self.raise_on_construct
        pub = _FakeChannelPublisher(topic, message_class)
        self.built.append(pub)
        return pub


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> _RecordingChannelFactory:
    """Install a fake ``unitree_sdk2py.core.channel`` module for the test.

    The publisher's lazy import path is ``from unitree_sdk2py.core.channel
    import ChannelPublisher`` - so this fixture makes exactly that import
    resolve to a recording factory, without adding the real SDK as a test
    dependency.
    """
    factory = _RecordingChannelFactory()
    fake_module = types.ModuleType("unitree_sdk2py.core.channel")
    fake_module.ChannelPublisher = factory  # type: ignore[attr-defined]
    fake_parent = types.ModuleType("unitree_sdk2py.core")
    fake_grandparent = types.ModuleType("unitree_sdk2py")
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", fake_grandparent)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", fake_parent)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", fake_module)
    return factory


@pytest.fixture
def dds_ready(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``ensure_dds`` succeed without touching a real interface.

    ``ensure_dds`` normally calls ``ChannelFactoryInitialize`` from the SDK.
    Under the fake SDK a bare call still fails (the fake has no
    ``ChannelFactoryInitialize``) so tests that need a started publisher use
    this fixture, which short-circuits ``ensure_dds`` and clears the state
    on teardown.
    """
    monkeypatch.setattr(
        "strands_robots.tools.g1._g1_common.ensure_dds",
        lambda interface: None,
    )
    monkeypatch.setattr(
        "strands_robots.tools.g1._dds_engine.ensure_dds",
        lambda interface: None,
    )
    yield
    reset_dds_state()


# =========================================================================
# The class exists and does what its docstring says.                      #
# =========================================================================


def test_ddspublisher_is_importable_without_the_sdk() -> None:
    """``import DDSPublisher`` must not touch ``unitree_sdk2py``.

    The whole point of the class is that the driver module can be imported
    on a headless box. If someone regresses this by adding a module-level
    ``import unitree_sdk2py``, the import here will fail on Thor and CI
    before the driver ever loads.
    """
    # The class was already imported at test module load; this asserts the
    # module did not stash a live SDK handle at import time either.
    engine_ns = vars(_dds_engine)
    assert "ChannelPublisher" not in engine_ns
    assert "unitree_sdk2py" not in engine_ns


def test_start_is_idempotent(dds_ready: None) -> None:
    """A second :meth:`start` on the same interface is a no-op.

    Mirrors :class:`DDSSubscriberSet` - the driver may build both classes
    from the same ``connect_eagerly`` call, and both may see a re-entrant
    caller.
    """
    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    assert pub_set.start() is None


def test_get_publisher_before_start_refuses(fake_sdk: _RecordingChannelFactory) -> None:
    """A publisher cannot be built before :meth:`start` succeeds.

    Same discipline as the subscriber set: you cannot subscribe to a bus
    that has not been initialised. The error string is stable so callers
    can grep for it.
    """
    pub_set = DDSPublisher("eth0")
    pub, err = pub_set.get_publisher("rt/lowcmd", _FakeChannelPublisher)
    assert pub is None
    assert err == "DDS not initialised - call start() first"
    assert fake_sdk.built == []


def test_get_publisher_constructs_once_per_key(
    fake_sdk: _RecordingChannelFactory,
    dds_ready: None,
) -> None:
    """Two calls with the same ``(topic, message_class)`` share a publisher.

    A control loop at 500Hz asks for the ``rt/lowcmd`` publisher every step;
    the cache is the difference between one DDS construction and 500 per
    second.
    """

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    p1, err1 = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    p2, err2 = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    assert err1 is None and err2 is None
    assert p1 is p2
    assert len(fake_sdk.built) == 1
    assert fake_sdk.built[0].topic == "rt/lowcmd"
    assert fake_sdk.built[0].message_class is LowCmdStub
    assert fake_sdk.built[0].init_calls == 1


def test_get_publisher_two_topics_two_publishers(
    fake_sdk: _RecordingChannelFactory,
    dds_ready: None,
) -> None:
    """Different topics get different publishers, same interface."""

    class LowCmdStub:
        pass

    class ArmSdkStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    p_low, _ = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    p_arm, _ = pub_set.get_publisher("rt/armsdk", ArmSdkStub)
    assert p_low is not p_arm
    assert len(fake_sdk.built) == 2


def test_publish_writes_the_message(
    fake_sdk: _RecordingChannelFactory,
    dds_ready: None,
) -> None:
    """The message passed to :meth:`publish` reaches the publisher's ``Write``."""

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    msg = LowCmdStub()
    assert pub_set.publish("rt/lowcmd", LowCmdStub, msg) is None
    assert len(fake_sdk.built) == 1
    assert fake_sdk.built[0].writes == [msg]


def test_publish_returns_error_when_write_raises(
    fake_sdk: _RecordingChannelFactory,
    dds_ready: None,
) -> None:
    """A ``Write`` that raises is caught and turned into a string error.

    Same envelope as the subscriber set's ``subscribe`` on a failed
    constructor - the driver never raises to its caller from a publish.
    """

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    # Force the first write to raise.
    pub, _ = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    assert pub is not None
    pub.write_should_raise = RuntimeError("bus overrun")
    err = pub_set.publish("rt/lowcmd", LowCmdStub, LowCmdStub())
    assert err is not None
    assert "rt/lowcmd" in err
    assert "bus overrun" in err


def test_publish_before_start_refuses(fake_sdk: _RecordingChannelFactory) -> None:
    """A caller that skips ``start`` gets the same refusal as ``get_publisher``."""

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    err = pub_set.publish("rt/lowcmd", LowCmdStub, LowCmdStub())
    assert err == "DDS not initialised - call start() first"


def test_publisher_construction_error_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    dds_ready: None,
) -> None:
    """A ``ChannelPublisher(...)`` that raises is caught and named."""
    fake_module = types.ModuleType("unitree_sdk2py.core.channel")
    fake_module.ChannelPublisher = _RecordingChannelFactory(  # type: ignore[attr-defined]
        raise_on_construct=OSError("no interface"),
    )
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core", types.ModuleType("unitree_sdk2py.core"))
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", fake_module)

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    pub, err = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    assert pub is None
    assert err is not None
    assert "rt/lowcmd" in err
    assert "no interface" in err


def test_close_is_idempotent_and_drops_cache(
    fake_sdk: _RecordingChannelFactory,
    dds_ready: None,
) -> None:
    """After :meth:`close`, the next :meth:`get_publisher` reconstructs.

    Mirrors :class:`DDSSubscriberSet.close`. A driver going through a
    disconnect/reconnect must not silently keep a stale handle.
    """

    class LowCmdStub:
        pass

    pub_set = DDSPublisher("eth0")
    assert pub_set.start() is None
    p1, _ = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    pub_set.close()
    pub_set.close()  # idempotent
    p2, _ = pub_set.get_publisher("rt/lowcmd", LowCmdStub)
    assert p1 is not p2
    assert len(fake_sdk.built) == 2


# =========================================================================
# The invariants shared with DDSSubscriberSet.                            #
# =========================================================================


def test_publisher_and_subscriber_share_the_init_lock() -> None:
    """Reading :data:`_DDS_INIT_LOCK` from both classes returns the same object.

    Otherwise the segfault the docstring warns about is reachable by a
    driver that constructs a subscriber and a publisher on two threads.
    """
    # The module-level import binds the same object; a copy would be a bug.
    from strands_robots.tools.g1 import _dds_engine as engine
    from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK as canonical

    # Both classes read the module-level name inside their methods; assert
    # the module binding is the canonical lock object.
    assert engine._DDS_INIT_LOCK is canonical


def test_module_does_not_import_the_sdk_at_load_time() -> None:
    """A grep-scale invariant: no top-level SDK import.

    Test lives here rather than a whole-tree grep because the promise is
    specifically the DDSPublisher's own promise - it is what makes headless
    tests possible.
    """
    import inspect

    source = inspect.getsource(_dds_engine)
    # The lazy imports are inside function bodies; a module-level import
    # would be at column 0.
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import unitree_sdk2py") or stripped.startswith("from unitree_sdk2py"):
            # It is only OK if it is indented (i.e. inside a function).
            assert line != stripped, f"module-level SDK import: {line!r}"
