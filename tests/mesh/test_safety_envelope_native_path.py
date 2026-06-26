"""Zenoh-native-path coverage for the safety-envelope publisher.

The legacy fallback path (no Zenoh session available) is pinned in
``test_safety_envelope_fallback``. These tests pin the complementary
Zenoh-native path that runs when a TLS-bound session *is* open:

* ``_local_session_zid`` resolving (or refusing) the per-session ZID,
* ``_safety_publisher_for`` lazily declaring and then *reusing* a
  publisher (and degrading to ``None`` when declaration fails),
* ``_next_safety_sn`` handing out monotonic, per-key sequence numbers,
* ``_publish_safety_envelope`` attaching a ``SourceInfo`` to the put and
  falling back to the legacy ``put`` when the native put raises.

These are the wire-attribution / replay-defence contracts that the
receiver relies on, so they are asserted behaviourally (what gets put on
the wire), not by inspecting private state.
"""

import sys
import types

from strands_robots.mesh import core
from strands_robots.mesh import session as session_mod


class _FakeInfo:
    def __init__(self, zid):
        self._zid = zid

    def zid(self):
        if isinstance(self._zid, Exception):
            raise self._zid
        return self._zid


class _FakePublisher:
    def __init__(self, key):
        self.key = key
        self.id = f"eid::{key}"
        self.puts: list[tuple[bytes, object]] = []
        self.put_error: Exception | None = None

    def put(self, payload, source_info=None):
        if self.put_error is not None:
            raise self.put_error
        self.puts.append((payload, source_info))


class _FakeSession:
    def __init__(self, zid="ab12cd34", declare_error=None):
        self.info = _FakeInfo(zid)
        self._declare_error = declare_error
        self.declared: list[_FakePublisher] = []

    def declare_publisher(self, key):
        if self._declare_error is not None:
            raise self._declare_error
        pub = _FakePublisher(key)
        self.declared.append(pub)
        return pub


def _bind_session(monkeypatch, session):
    """Point core's lazy session lookup at *session* (or None)."""
    monkeypatch.setattr(session_mod, "_current_zenoh_session_directly", lambda: session)


def _install_fake_zenoh(monkeypatch, with_source_info=True):
    fake = types.ModuleType("zenoh")
    if with_source_info:

        class _SourceInfo:
            def __init__(self, source_id, sn):
                self.source_id = source_id
                self.sn = sn

        fake.SourceInfo = _SourceInfo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zenoh", fake)
    return fake


# --- _local_session_zid -----------------------------------------------------


def test_local_session_zid_returns_stringified_zid(monkeypatch):
    _bind_session(monkeypatch, _FakeSession(zid="deadbeef"))
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._local_session_zid() == "deadbeef"


def test_local_session_zid_none_when_no_session(monkeypatch):
    _bind_session(monkeypatch, None)
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._local_session_zid() is None


def test_local_session_zid_none_when_zid_call_raises(monkeypatch):
    _bind_session(monkeypatch, _FakeSession(zid=RuntimeError("not ready")))
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._local_session_zid() is None


def test_local_session_zid_none_when_zid_is_none(monkeypatch):
    _bind_session(monkeypatch, _FakeSession(zid=None))
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._local_session_zid() is None


# --- _safety_publisher_for --------------------------------------------------


def test_safety_publisher_for_declares_once_and_reuses(monkeypatch):
    session = _FakeSession()
    _bind_session(monkeypatch, session)
    m = core.Mesh(robot=object(), peer_id="t1")

    first = m._safety_publisher_for("strands/safety/estop")
    second = m._safety_publisher_for("strands/safety/estop")

    assert first is second
    # Reuse: the session only ever declared one publisher for this key.
    assert len(session.declared) == 1


def test_safety_publisher_for_none_when_declare_fails(monkeypatch):
    session = _FakeSession(declare_error=RuntimeError("declare boom"))
    _bind_session(monkeypatch, session)
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._safety_publisher_for("strands/safety/estop") is None


def test_safety_publisher_for_none_when_no_session(monkeypatch):
    _bind_session(monkeypatch, None)
    m = core.Mesh(robot=object(), peer_id="t1")
    assert m._safety_publisher_for("strands/safety/estop") is None


# --- _next_safety_sn --------------------------------------------------------


def test_next_safety_sn_is_monotonic_and_per_key(monkeypatch):
    _bind_session(monkeypatch, None)
    m = core.Mesh(robot=object(), peer_id="t1")

    assert m._next_safety_sn("a") == 1
    assert m._next_safety_sn("a") == 2
    # Independent counter for a different key.
    assert m._next_safety_sn("b") == 1
    assert m._next_safety_sn("a") == 3


# --- _publish_safety_envelope (native path) ---------------------------------


def test_publish_native_attaches_source_info(monkeypatch):
    session = _FakeSession()
    _bind_session(monkeypatch, session)
    _install_fake_zenoh(monkeypatch)

    legacy_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(core, "put", lambda k, p: legacy_calls.append((k, p)))

    m = core.Mesh(robot=object(), peer_id="t1")
    m._publish_safety_envelope(
        "strands/safety/estop",
        {"peer_id": "t1", "t": 1.0, "source_zid": "ab12cd34"},
    )

    # Native path used: the declared publisher carries the put, with a
    # SourceInfo whose sequence number is the first allocation for the key.
    assert len(session.declared) == 1
    pub = session.declared[0]
    assert len(pub.puts) == 1
    payload_bytes, source_info = pub.puts[0]
    assert source_info is not None
    assert source_info.source_id == pub.id
    assert source_info.sn == 1
    # The legacy transport-agnostic put is NOT used on the native path.
    assert legacy_calls == []
    # Body is published intact (source_zid retained on the native path).
    import json

    decoded = json.loads(payload_bytes.decode())
    assert decoded["source_zid"] == "ab12cd34"


def test_publish_native_put_failure_falls_back_to_legacy(monkeypatch):
    session = _FakeSession()
    _bind_session(monkeypatch, session)
    _install_fake_zenoh(monkeypatch)

    legacy_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(core, "put", lambda k, p: legacy_calls.append((k, p)))

    m = core.Mesh(robot=object(), peer_id="t1")
    # Pre-declare the publisher so we can arm its put to fail.
    pub = m._safety_publisher_for("strands/safety/estop")
    assert pub is not None
    pub.put_error = RuntimeError("link down")

    m._publish_safety_envelope(
        "strands/safety/estop",
        {"peer_id": "t1", "t": 1.0, "source_zid": "ab12cd34"},
    )

    # Native put raised -> legacy put used, with the wire zid stripped.
    assert len(legacy_calls) == 1
    key, payload = legacy_calls[0]
    assert key == "strands/safety/estop"
    assert "source_zid" not in payload


def test_publish_native_without_source_info_ctor_falls_back(monkeypatch):
    session = _FakeSession()
    _bind_session(monkeypatch, session)
    # zenoh present but lacking the SourceInfo constructor (very old build).
    _install_fake_zenoh(monkeypatch, with_source_info=False)

    legacy_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(core, "put", lambda k, p: legacy_calls.append((k, p)))

    m = core.Mesh(robot=object(), peer_id="t1")
    m._publish_safety_envelope(
        "strands/safety/estop",
        {"peer_id": "t1", "t": 1.0, "source_zid": "ab12cd34"},
    )

    assert len(legacy_calls) == 1
    key, payload = legacy_calls[0]
    assert key == "strands/safety/estop"
    assert "source_zid" not in payload
