"""Repro: Robot(mesh=True) without eclipse-zenoh installed emits the ACL-refusal
message and NEVER the "zenoh is not installed" one, sending fresh users on a
wild goose chase configuring ACL when the root cause is a missing dep.

Steps
-----
    $ uv venv --python 3.12 && source .venv/bin/activate
    $ uv pip install "strands-robots[sim-mujoco]"   # note: NO [mesh] extra
    $ python bugbash/mesh_zenoh_error_order_repro.py

Expected (post-fix)
-------------------
    On the first ``Robot(mesh=True)`` call, the captured stderr MENTIONS
    zenoh once it is absent, e.g.::

      "eclipse-zenoh is not installed, so the mesh stays off. Install it with:
       pip install 'strands-robots[mesh]'"

    and does NOT print the multi-line ACL-refusal error, which is irrelevant
    when there is no transport to guard.

Actual (v0.5.2, main @ b43d96e)
-------------------------------
    The ACL-refusal is printed (misleading), and the zenoh-missing warning is
    never emitted, because ``Mesh.start()`` returns after
    ``_refuse_under_permissive_default_acl()`` and never reaches
    ``get_session()`` (which is where ``_report_zenoh_missing()`` fires).

    See ``strands_robots/mesh/core.py:851-865`` and
    ``strands_robots/mesh/session.py:1065``.
"""

from __future__ import annotations

import io
import logging
import os
import sys


def main() -> int:
    # Wipe any inherited STRANDS_MESH_* so we reproduce a fresh-user posture.
    for k in list(os.environ):
        if k.startswith("STRANDS_MESH"):
            del os.environ[k]

    # Sanity: zenoh must NOT be importable for this repro.
    try:
        import zenoh  # noqa: F401
    except ImportError:
        pass
    else:
        print("SKIP: eclipse-zenoh is installed in this env; uninstall to reproduce")
        return 0

    # Attach a StringIO handler to the root logger to capture every WARNING/
    # ERROR the mesh subsystem emits, then restore.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    try:
        from strands_robots import Robot
        r = Robot("so101", mesh=True)  # noqa: F841 - kept alive for lifecycle
    finally:
        root.removeHandler(handler)

    output = buf.getvalue()
    print("---- captured mesh/safety log output ----")
    print(output)
    print("---- end ----")

    mentions_zenoh = "zenoh" in output.lower()
    mentions_install_mesh = "pip install" in output.lower() and "[mesh]" in output.lower()
    mentions_acl = "acl" in output.lower() or "ACCEPT_PERMISSIVE_ACL" in output

    print(f"mentions_zenoh          = {mentions_zenoh}")
    print(f"mentions_install_[mesh] = {mentions_install_mesh}")
    print(f"mentions_acl            = {mentions_acl}")

    if not mentions_zenoh:
        print(
            "\nFAIL: user is not told zenoh is missing; the ACL-refusal banner is "
            "misleading in this posture.",
            file=sys.stderr,
        )
        return 1
    print("\nOK: fresh-user is directed at the real root cause.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
