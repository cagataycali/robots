# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A preset 3DGS scene download bounds every socket operation.

:func:`strands_robots.rendering.download_gsplat_scene` fetched its asset with
``urllib.request.urlretrieve``, whose signature is ``url, filename, reporthook,
data`` - there is no timeout parameter to pass. The fetch therefore ran on the
process-global default socket timeout, which is ``None`` unless some unrelated
import mutated it, so a peer that completed the connection and then stopped
sending held the calling thread indefinitely.

That is not merely slow, it defeats a documented fail-loud contract. The
``isaac_gs`` demo's ``resolve_background`` raises ``RuntimeError`` when the
photoreal path cannot initialize - a download failure included - precisely so a
user who asked for the captured scene never silently receives the procedural
gradient, and ``allow_fallback=True`` restores that demotion "for every failure
mode". A stalled peer produces neither outcome: nothing raises, so nothing is
demoted, and the fallback flag whose whole purpose is to keep the demo
rendering cannot fire either. The equivalent Gradio handler in the
``mujoco_gs`` demo wedges after yielding its "Downloading ..." progress line.

These tests pin the corrected contract:

* a peer that stops mid-body raises inside the stated bound and leaves the
  cache path empty, measured against a real stalled listener rather than a
  double, with the assertion made on a worker thread so an unbounded fetch
  renders as a failure instead of hanging the suite;
* the bound is a caller-stated positive finite number of seconds, graded by the
  same domain the rest of the tree uses for a span of time, and the default is
  itself finite;
* the cache path is never handed a partial body - a body short of the peer's
  declared ``Content-Length`` fails the fetch and leaves nothing to read;
* no module in the package fetches through ``urlretrieve``, which cannot be
  bounded at all.

No network access: the stalled peer is a loopback socket, and the transport
doubles stand in for ``urlopen`` everywhere else.
"""

from __future__ import annotations

import ast
import math
import socket
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from strands_robots.rendering import backgrounds, download_gsplat_scene

#: A preset whose source URL ends in ``.ply``; its slug is "bonsai".
PRESET = "bonsai (indoor tabletop)"

#: Bound handed to the fetch in the stalled-peer test. Small enough to keep the
#: test quick, and the peer never sends again, so the only ways out are the
#: bound firing or the thread never finishing.
STATED_BOUND_S = 0.5

#: How long the assertion waits for the fetch to come back. Twenty times the
#: stated bound, so a bounded fetch cannot fail this for being slow while an
#: unbounded one still reports promptly.
GRACE_S = STATED_BOUND_S * 20


@pytest.fixture
def stalled_peer():
    """A loopback listener that answers with headers, then stops sending.

    This is the failure a bound exists for: the connection is established and
    the response line and headers arrive, so nothing about the transport looks
    broken, and then the body never comes. Yields the URL to fetch.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    release = threading.Event()
    connections: list[socket.socket] = []

    def serve() -> None:
        try:
            conn, _ = listener.accept()
        except OSError:  # pragma: no cover - listener closed before a client came
            return
        connections.append(conn)
        try:
            conn.recv(65536)
            # A complete, entirely plausible response head promising a body...
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 4194304\r\n\r\n")
            conn.sendall(b"PLY-PREFIX")
            # ... and then silence, until teardown lets the thread go.
            release.wait(120)
        except OSError:  # pragma: no cover - client hung up first
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/scene.ply"
    finally:
        # Releasing the peer closes the socket, which unblocks any fetch still
        # waiting on it, so a failing (unbounded) run leaves nothing wedged.
        release.set()
        for conn in connections:
            conn.close()
        listener.close()
        thread.join(GRACE_S)


def _fetch_on_a_thread(**kwargs: Any) -> tuple[threading.Thread, list[Exception], list[Path]]:
    """Run the fetch on a worker thread, capturing its outcome.

    An unbounded fetch never returns, and a test that calls it directly hangs
    the suite instead of failing it. Running it here turns "never came back"
    into an assertion the caller can make.
    """
    raised: list[Exception] = []
    returned: list[Path] = []

    def attempt() -> None:
        try:
            returned.append(download_gsplat_scene(**kwargs))
        except Exception as exc:  # noqa: BLE001 - which exception it is IS the measurement
            raised.append(exc)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    return thread, raised, returned


class TestAStalledPeerDoesNotHoldTheCaller:
    def test_a_peer_that_stops_mid_body_raises_inside_the_stated_bound(
        self, tmp_path, monkeypatch, stalled_peer
    ) -> None:
        monkeypatch.setitem(backgrounds.GSPLAT_SCENES, PRESET, stalled_peer)

        thread, raised, returned = _fetch_on_a_thread(name=PRESET, cache_dir=tmp_path, timeout=STATED_BOUND_S)
        thread.join(GRACE_S)

        assert not thread.is_alive(), (
            f"the fetch did not come back within {GRACE_S}s against a peer that stopped "
            f"sending, having been given a {STATED_BOUND_S}s bound: the bound never "
            "reached the socket, so the caller waits on the peer indefinitely"
        )
        assert not returned, "a peer that never sent the body must not yield a cached scene"
        assert isinstance(raised[0], TimeoutError), f"expected a timeout, got {raised[0]!r}"
        # The bound firing is only useful if the next call refetches: the ten
        # bytes the peer did send must not be sitting at the cache path, where
        # every later call would read them as the whole scene.
        assert list(tmp_path.iterdir()) == [tmp_path / "bonsai.ply.part"]


class TestTheBoundIsACallerStatedDomain:
    @pytest.mark.parametrize(
        "unusable",
        [0, 0.0, -1.0, math.inf, -math.inf, math.nan, True, None, "5"],
        ids=["zero", "zero-float", "negative", "inf", "-inf", "nan", "bool", "none", "str"],
    )
    def test_a_bound_that_bounds_nothing_is_refused(self, tmp_path, unusable) -> None:
        # A bound of 0 or a negative one is not a deadline, inf states no
        # deadline, nan loses every comparison it reaches, and True would act
        # as a silent 1 second. Refused before any socket is opened.
        with pytest.raises(ValueError, match="timeout"):
            download_gsplat_scene(PRESET, cache_dir=tmp_path, timeout=unusable)
        assert list(tmp_path.iterdir()) == []

    def test_an_unknown_preset_is_still_refused_before_the_bound_is_read(self, tmp_path) -> None:
        # Naming the scene is the caller's first mistake to hear about; the
        # timeout domain must not preempt it.
        with pytest.raises(KeyError, match="Unknown scene"):
            download_gsplat_scene("not-a-scene", cache_dir=tmp_path, timeout=0)

    def test_a_fractional_bound_reaches_the_transport(self, tmp_path, monkeypatch) -> None:
        seen = _record_bounds(monkeypatch)
        download_gsplat_scene(PRESET, cache_dir=tmp_path, timeout=0.25)
        assert seen == [0.25]

    def test_the_default_bound_is_finite_and_positive(self, tmp_path, monkeypatch) -> None:
        # A caller who states nothing must still get a deadline: the default is
        # what every existing call site in the tree and the demos rely on.
        seen = _record_bounds(monkeypatch)
        download_gsplat_scene(PRESET, cache_dir=tmp_path)
        assert len(seen) == 1
        assert isinstance(seen[0], float)
        assert 0 < seen[0] < math.inf


class TestTheCachePathNeverSeesAPartialBody:
    def test_a_body_short_of_the_declared_length_fails_the_fetch(self, tmp_path, monkeypatch) -> None:
        # urlretrieve raised ContentTooShortError for this; losing that check
        # would trade a hang for a truncated scene cached as a complete one.
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=None: _body(b"short", declared="4194304"),
        )
        with pytest.raises(urllib.error.ContentTooShortError):
            download_gsplat_scene(PRESET, cache_dir=tmp_path)
        assert not (tmp_path / "bonsai.ply").exists()

    def test_a_peer_that_declares_no_usable_length_is_taken_at_its_word(self, tmp_path, monkeypatch) -> None:
        # A chunked response carries no Content-Length, and a non-numeric one
        # is not a truncation signal in either direction. Neither is a reason
        # to refuse a body that arrived.
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=None: _body(b"ply-bytes", declared=None),
        )
        assert download_gsplat_scene(PRESET, cache_dir=tmp_path).read_bytes() == b"ply-bytes"

        (tmp_path / "bonsai.ply").unlink()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=None: _body(b"ply-bytes", declared="not-a-number"),
        )
        assert download_gsplat_scene(PRESET, cache_dir=tmp_path).read_bytes() == b"ply-bytes"

    def test_a_body_longer_than_declared_is_not_refused(self, tmp_path, monkeypatch) -> None:
        # Only a SHORT body is evidence of truncation; a peer that undersold
        # the length still delivered everything it sent.
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=None: _body(b"ply-bytes", declared="2"),
        )
        assert download_gsplat_scene(PRESET, cache_dir=tmp_path).read_bytes() == b"ply-bytes"


class TestNoFetchInThePackageIsUnboundable:
    def test_no_module_downloads_through_urlretrieve(self) -> None:
        """``urlretrieve`` takes no timeout, so any call to it is unbounded.

        Unlike a missing ``timeout=`` argument, this is not a call that forgot
        to state a bound - it is a call that cannot state one, so the whole
        package is held to the stronger rule.
        """
        package = Path(backgrounds.__file__).resolve().parent.parent
        scanned = 0
        offenders: list[str] = []
        for module in sorted(package.rglob("*.py")):
            scanned += 1
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "urlretrieve":
                    offenders.append(f"{module.relative_to(package)}:{node.lineno}")
                elif isinstance(node, ast.Name) and node.id == "urlretrieve":
                    offenders.append(f"{module.relative_to(package)}:{node.lineno}")
        # The rule is only worth anything if the scan actually reached the tree.
        assert scanned > 100, f"the scan visited only {scanned} modules; the roster rule is vacuous"
        assert offenders == []


def _record_bounds(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Stub the transport, returning the list of bounds it is handed."""
    seen: list[float | None] = []

    def _recording_urlopen(url: str, timeout: float | None = None) -> Any:
        seen.append(timeout)
        return _body(b"ply-bytes")

    monkeypatch.setattr("urllib.request.urlopen", _recording_urlopen)
    return seen


def _body(payload: bytes, declared: str | None = "") -> Any:
    """A urlopen double serving *payload*, declaring *declared* bytes.

    ``declared=""`` (the default) means "declare the real length"; ``None``
    means send no ``Content-Length`` at all.
    """

    class _Response:
        def __init__(self) -> None:
            self._remaining = payload
            if declared is None:
                self.headers: dict[str, str] = {}
            else:
                self.headers = {"Content-Length": declared or str(len(payload))}

        def read(self, amount: int | None = None) -> bytes:
            chunk, self._remaining = self._remaining, b""
            return chunk

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    return _Response()
