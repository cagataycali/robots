"""
Repro: PolicyServer.start()/.serve() gives a bare ``ModuleNotFoundError: No module
named 'websockets'`` on a ``pip install strands-robots`` env (no ``[inference]``
extra), while the sibling client ``RemotePolicy`` reports an actionable
``require_held_connect("RemotePolicy", "inference")`` message.

Asymmetric guard between the two halves of the client/server split
documented at ``docs/learn/policies/remote.md``:

- client (``strands_robots/inference/client.py:225``)::

      require_held_connect(type(self).__name__, "inference")

  => raises:
      "RemotePolicy needs the websockets package, which is not installed:
       pip install 'strands-robots[inference]'"
      (ImportError, name='websockets')

- server (``strands_robots/inference/server.py:306, :364``)::

      from websockets.sync.server import serve

  => raises:
      "No module named 'websockets'"
      (ModuleNotFoundError, name=None in the message)

A reader who follows the quickstart from docs/learn/policies/remote.md literally::

    from strands_robots.inference import PolicyServer
    PolicyServer(policy_provider="mock", port=0).start()

on the GPU host (the host that needs the EXTRA, because the docs explicitly
tell them "the robot host installs websockets") lands at the bare error.
The pointer to ``strands-robots[inference]`` is not surfaced.
"""

from __future__ import annotations

import importlib.abc
import sys


class _BlockWebsockets(importlib.abc.MetaPathFinder):
    """A meta-path finder that refuses to import ``websockets`` or its submodules.

    Simulates a ``pip install strands-robots`` with no extras: the base
    distribution does NOT declare websockets, so a venv built that way has
    no module ``websockets`` available.
    """

    def find_spec(self, name, path, target=None):
        if name == "websockets" or name.startswith("websockets."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


def _install_blocker() -> None:
    sys.meta_path.insert(0, _BlockWebsockets())
    # Purge any already-imported websockets modules the dev env may hold.
    for key in list(sys.modules):
        if key == "websockets" or key.startswith("websockets."):
            del sys.modules[key]


def main() -> int:
    _install_blocker()

    # Half A: the server side (what docs/learn/policies/remote.md tells the
    # GPU-host operator to run first).
    print("=" * 68)
    print("A) PolicyServer.start()  --  no [inference] extra:")
    print("=" * 68)
    from strands_robots.inference import PolicyServer

    try:
        PolicyServer(policy_provider="mock", port=0).start()
        print("UNEXPECTED: start() returned")
        return 1
    except ModuleNotFoundError as exc:
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  name attribute: {getattr(exc, 'name', None)!r}")
        print(f"  message mentions 'strands-robots[inference]': "
              f"{'strands-robots[inference]' in str(exc)}")
        print(f"  message mentions pip install: {'pip install' in str(exc)}")
    except ImportError as exc:
        # An actionable gate would raise ImportError (not ModuleNotFoundError
        # straight from the import machinery) with a strands-robots[inference]
        # pointer, matching the client half below.
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  message mentions 'strands-robots[inference]': "
              f"{'strands-robots[inference]' in str(exc)}")

    # Half B: the client side (symmetric call site, same extra, same floor).
    print()
    print("=" * 68)
    print("B) RemotePolicy()        --  no [inference] extra:")
    print("=" * 68)
    # Re-import under the blocker to make sure the comparison is apples-to-apples.
    for key in list(sys.modules):
        if key == "websockets" or key.startswith("websockets."):
            del sys.modules[key]

    from strands_robots.policies import create_policy

    try:
        create_policy("remote", endpoint="ws://localhost:8765")
        print("UNEXPECTED: create_policy returned")
        return 1
    except ImportError as exc:
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  name attribute: {getattr(exc, 'name', None)!r}")
        print(f"  message mentions 'strands-robots[inference]': "
              f"{'strands-robots[inference]' in str(exc)}")

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print("Before the fix:")
    print("  Server half: bare ModuleNotFoundError, no mention of the extra.")
    print("  Client half: ImportError, names strands-robots[inference].")
    print("After the fix (require_held_connect at start/serve):")
    print("  Both halves: ImportError, naming strands-robots[inference].")
    print("The gate is the one chokepoint that closes the asymmetric guard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
