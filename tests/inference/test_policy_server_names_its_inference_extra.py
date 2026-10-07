"""``PolicyServer.start()``/``serve()`` report the extra, like ``RemotePolicy``.

Pins the fix for the asymmetric guard between the two halves of the
client/server split documented at ``docs/learn/policies/remote.md``:

- ``RemotePolicy._connect`` (``inference/client.py:225``) refuses a base
  install with ``require_held_connect("RemotePolicy", "inference")``, so a
  caller without the ``[inference]`` extra reads::

      RemotePolicy needs the websockets package, which is not installed:
      pip install 'strands-robots[inference]'

- ``PolicyServer.start`` / ``PolicyServer.serve`` historically ran a bare
  ``from websockets.sync.server import serve`` as their first line, which on
  the same base install raised::

      ModuleNotFoundError: No module named 'websockets'

  naming neither the extra nor this package. The fix mirrors the client half,
  calling ``require_held_connect(type(self).__name__, "inference")`` first.

The test blocks the real ``websockets`` through ``sys.meta_path`` so that the
report that reaches the caller is the gate's, not the dev env's installed-but-
old ``websockets``.
"""

from __future__ import annotations

import importlib.abc
import sys

import pytest


class _BlockWebsockets(importlib.abc.MetaPathFinder):
    """Refuses to import ``websockets`` or its submodules. Simulates a
    ``pip install strands-robots`` with no ``[inference]`` extra.
    """

    def find_spec(self, name, path, target=None):
        if name == "websockets" or name.startswith("websockets."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


@pytest.fixture
def no_websockets(monkeypatch):
    """Make ``websockets`` look uninstalled for the duration of one test.

    Blocks ``import websockets`` AND makes
    ``importlib.metadata.version("websockets")`` raise ``PackageNotFoundError``
    - a real base install has neither the module importable nor its
    distribution metadata on disk, and the gate reads both.
    """
    from importlib import metadata

    blocker = _BlockWebsockets()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    # Scrub any cached websockets modules the dev env may hold; the real
    # one lives at a lower meta_path finder and would otherwise shadow the
    # blocker if its spec is already in sys.modules.
    for key in list(sys.modules):
        if key == "websockets" or key.startswith("websockets."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    # Simulate the distribution missing too, so ``metadata.version`` raises
    # ``PackageNotFoundError`` just as it would on a fresh venv that never
    # installed the extra.
    real_version = metadata.version

    def _spoof_version(name: str, *args, **kwargs) -> str:
        if name == "websockets":
            raise metadata.PackageNotFoundError(name)
        return real_version(name, *args, **kwargs)

    monkeypatch.setattr(metadata, "version", _spoof_version)
    # The gate imports ``metadata`` as a module-level attribute of _ws_wire,
    # so patch that reference too.
    import strands_robots.policies._ws_wire as _ws_wire

    monkeypatch.setattr(_ws_wire.metadata, "version", _spoof_version)
    # Also drop the server module so our fresh import re-reaches the gate
    # (the gate runs on every call, so this is belt-and-suspenders).
    monkeypatch.delitem(sys.modules, "strands_robots.inference.server", raising=False)


class TestStartNamesTheInferenceExtra:
    """``PolicyServer.start()`` refuses a websockets-less env with the extra."""

    def test_start_reports_strands_robots_inference(self, no_websockets):
        from strands_robots.inference import PolicyServer

        with pytest.raises(ImportError) as excinfo:
            PolicyServer(policy_provider="mock", port=0).start()

        exc = excinfo.value
        assert "strands-robots[inference]" in str(exc), (
            "PolicyServer.start() must point at the [inference] extra, "
            "mirroring RemotePolicy._connect; got: " + str(exc)
        )
        assert exc.name == "websockets", (
            "ImportError.name must be 'websockets' so a caller can "
            "distinguish an absent optional dep from a broken package path"
            " without parsing the message; got: " + repr(exc.name)
        )

    def test_start_does_not_raise_the_bare_module_not_found_error(
        self, no_websockets
    ):
        """The gate replaces ``ModuleNotFoundError: No module named 'websockets'``.

        A caller catching ``ImportError`` is fine either way, but the test
        pins the SPECIFIC subclass: the gate is an :class:`ImportError`, not
        a :class:`ModuleNotFoundError` straight from the import machinery, so
        the message carries the strands-robots pointer rather than naming
        only the module Python could not find.
        """
        from strands_robots.inference import PolicyServer

        with pytest.raises(ImportError) as excinfo:
            PolicyServer(policy_provider="mock", port=0).start()

        # Not a bare ``ModuleNotFoundError`` with the default message.
        assert str(excinfo.value) != "No module named 'websockets'", (
            "The refusal must come from require_held_connect, not from the "
            "bare import of websockets. If this assertion fails, the gate "
            "at the top of start() was removed or short-circuited."
        )


class TestServeNamesTheInferenceExtra:
    """``PolicyServer.serve()`` refuses a websockets-less env with the extra."""

    def test_serve_reports_strands_robots_inference(self, no_websockets):
        from strands_robots.inference import PolicyServer

        with pytest.raises(ImportError) as excinfo:
            PolicyServer(policy_provider="mock", port=0).serve()

        exc = excinfo.value
        assert "strands-robots[inference]" in str(exc), (
            "PolicyServer.serve() (standalone-server entry) must point at "
            "the [inference] extra, mirroring start(); got: " + str(exc)
        )
        assert exc.name == "websockets", (
            "ImportError.name must be 'websockets'; got: " + repr(exc.name)
        )


class TestSymmetryWithRemotePolicy:
    """The two halves of the client/server split report the same install."""

    def test_both_halves_name_the_same_extra(self, no_websockets):
        """Server and client are built against the same floor, so the
        remedy is identical. The gate pins that: a sentence that works
        for one half must work for the other.
        """
        from strands_robots.inference import PolicyServer
        from strands_robots.policies import create_policy

        with pytest.raises(ImportError) as server_exc:
            PolicyServer(policy_provider="mock", port=0).start()

        with pytest.raises(ImportError) as client_exc:
            create_policy("remote", endpoint="ws://localhost:8765")

        # Both messages contain the SAME install hint.
        hint = "pip install 'strands-robots[inference]'"
        assert hint in str(server_exc.value), (
            "server half missing the extra hint: " + str(server_exc.value)
        )
        assert hint in str(client_exc.value), (
            "client half missing the extra hint: " + str(client_exc.value)
        )
