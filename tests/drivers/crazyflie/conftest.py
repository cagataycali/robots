"""A hardware-shaped ``cflib`` stand-in, so the whole driver runs with no radio.

The fake is shaped like the SDK rather than like the driver: it exposes
``commander``, ``high_level_commander``, ``platform`` and ``log`` with the method
names ``cflib`` actually has, and it records **every call in order** on one
shared list. Ordering is the point - the driver's load-bearing sequences are
orderings (arm before any setpoint, ``send_notify_setpoint_stop`` before
``land``), and a per-object call list cannot see an ordering that spans two
objects.

Nothing here subclasses or imports ``cflib``: the tests must pass on a machine
that has never had a Crazyradio plugged in, which is every CI runner.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest


class _Recorder:
    """One shared, ordered log of ``(target.method, args)`` across every stub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._lock = threading.Lock()
        self._counts_reached = threading.Event()
        self._await_name: str | None = None
        self._await_count = 0

    def record(self, name: str, args: tuple[Any, ...]) -> None:
        with self._lock:
            self.calls.append((name, args))
            if self._await_name is not None and self.count(self._await_name) >= self._await_count:
                self._counts_reached.set()

    def count(self, name: str) -> int:
        """How many times ``name`` was called."""
        return sum(1 for called, _ in self.calls if called == name)

    def names(self) -> list[str]:
        """Just the call names, in order."""
        return [name for name, _ in self.calls]

    def args_of(self, name: str) -> tuple[Any, ...]:
        """Arguments of the first ``name`` call."""
        for called, args in self.calls:
            if called == name:
                return args
        raise AssertionError(f"{name} was never called; calls were {self.names()}")

    def wait_for(self, name: str, count: int, timeout: float = 5.0) -> bool:
        """Block until ``name`` has been called ``count`` times, or time out.

        Bounded, and returns whether the count was reached, so a test asserts on
        the outcome rather than on a sleep long enough to be flaky.
        """
        with self._lock:
            self._await_name, self._await_count = name, count
            if self.count(name) >= count:
                return True
            self._counts_reached.clear()
        return self._counts_reached.wait(timeout)


class _Stub:
    """Anything whose methods only need recording. Accepts any call."""

    def __init__(self, recorder: _Recorder, prefix: str, raises: BaseException | None = None) -> None:
        self._recorder = recorder
        self._prefix = prefix
        self._raises = raises

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> None:
            del kwargs
            self._recorder.record(f"{self._prefix}.{name}", args)
            if self._raises is not None:
                raise self._raises

        return call


class _FakeLogConfig:
    """Shaped like ``cflib.crazyflie.log.LogConfig``.

    ``data_received_cb.add_callback`` is how the driver subscribes, and holding
    the callback is what lets a test deliver a telemetry frame by hand.
    """

    def __init__(self, recorder: _Recorder, name: str = "", period_in_ms: int = 0) -> None:
        self._recorder = recorder
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables: list[tuple[str, str]] = []
        self.callbacks: list[Any] = []
        self.data_received_cb = _Callbacks(self.callbacks)

    def add_variable(self, name: str, fetch_as: str) -> None:
        self.variables.append((name, fetch_as))

    def start(self) -> None:
        self._recorder.record("log.start", ())

    def stop(self) -> None:
        self._recorder.record("log.stop", ())

    def deliver(self, data: dict[str, Any]) -> None:
        """Push one telemetry frame through, as the link thread would."""
        for callback in self.callbacks:
            callback(0, data, self)


class _Callbacks:
    def __init__(self, sink: list[Any]) -> None:
        self._sink = sink

    def add_callback(self, callback: Any) -> None:
        self._sink.append(callback)


class FakeCrazyflie:
    """Shaped like ``cflib.crazyflie.Crazyflie``."""

    def __init__(self, recorder: _Recorder, *, arming: BaseException | None = None) -> None:
        self.recorder = recorder
        self.commander = _Stub(recorder, "commander")
        self.high_level_commander = _Stub(recorder, "high_level")
        self.platform = _Stub(recorder, "platform", raises=arming)
        self.log = _FakeLink(recorder)
        self.uri: str | None = None

    def open_link(self, uri: str) -> None:
        self.uri = uri
        self.recorder.record("open_link", (uri,))

    def close_link(self) -> None:
        self.recorder.record("close_link", ())


class _FakeLink:
    """The ``cf.log`` surface: holds the one block the driver adds."""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.block: _FakeLogConfig | None = None

    def add_config(self, block: _FakeLogConfig) -> None:
        self.block = block
        self._recorder.record("log.add_config", (block.name,))


@pytest.fixture
def recorder() -> _Recorder:
    """The shared ordered call log every stub writes to."""
    return _Recorder()


@pytest.fixture
def connected(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder):  # type: ignore[no-untyped-def]
    """Build a connected, armed driver over the fake link.

    Returns a factory so a test can choose the constructor keywords (a faster
    ``setpoint_hz``, a failing arming request) and still get the same fake.
    """
    from strands_robots.drivers import crazyflie as module

    def build(*, arming: BaseException | None = None, **kwargs: Any):  # type: ignore[no-untyped-def]
        fake = FakeCrazyflie(recorder, arming=arming)
        pieces = type(
            "_Pieces",
            (),
            {
                "crtp": _Stub(recorder, "crtp"),
                "Crazyflie": lambda **_: fake,
                "LogConfig": lambda **kw: _FakeLogConfig(recorder, **kw),
            },
        )
        monkeypatch.setattr(module, "_resolve_cflib", lambda: pieces)
        driver = module.CrazyflieDriver(**kwargs)
        reason = driver.connect_eagerly()
        return driver, fake, reason

    return build
