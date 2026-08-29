"""Agent-facing wrapper for ``G1Driver.get_status``.

``G1Driver.get_status`` returns a JSON envelope naming the driver's
connection state, the last-observed FSM id, ``mode_machine``,
``battery_pct`` and the motion-switcher wire diagnostics
(``fsm_mode_name`` / ``fsm_refusal`` / ``motion_switcher_open_error``).
That envelope is what the mesh publishes on its status wire, but it is
not what an agent asks for: a caller planning a write wants to know
whether the arm-SDK gate would admit today, and the driver's answer is
"here is ``fsm_id``, compare it yourself against ``HANDSHAKE_FSMS`` /
``WALK_FSMS``". This verb closes that gap: it calls
:meth:`~strands_robots.drivers.g1.G1Driver.get_status`, then decides the
``admits_arm`` / ``admits_loco`` membership against the same driver
constants :mod:`~strands_robots.tools.g1.g1_motion_gates` names.

The verb takes a :class:`~strands_robots.drivers.g1.G1Driver` instance,
which is the first driver-instance-taking verb in this package. Every
earlier verb here (:mod:`~strands_robots.tools.g1.g1_joints`,
:mod:`~strands_robots.tools.g1.g1_motion_gates`) is a pure reader over
module-level constants and takes no argument; this one is a live read
against a wired driver and cannot answer without one. The driver
argument is typed :class:`~typing.Any` at runtime rather than as
``G1Driver``: the driver module imports ``ensure_dds`` from this package
at load, so a runtime import of ``G1Driver`` here would close a cycle,
and ``@tool`` calls :func:`typing.get_type_hints` at decoration time so
a string forward reference cannot resolve without pulling the driver at
import. The verb is duck-typed on ``get_status`` (any object with an
``async get_status`` returning the driver's envelope answers), which is
also how the tests hand it a hand-rolled double. ``import
strands_robots.tools.g1.g1_state`` still pulls no ``unitree_sdk2py``
submodule (the package's SDK-load-hygiene contract, refs
strands-labs/robots#358).

What this module does not do.

* Subscribe DDS. The driver's own subscribers deliver every field this
  verb returns; adding a second subscriber path would compete for the
  same topic and duplicate the bus load ``strands-labs/robots#358``'s
  singleton lock is meant to prevent.
* Rebuild a wedged loco RPC client. ``get_status`` reports what the
  driver's FSM refresher already read; a wedged RPC surfaces here as an
  ``fsm_id`` of ``None`` and an ``fsm_refusal`` string, decidably, and
  the recovery is on the driver's refresh loop rather than this read.
* Decode ``mode_machine`` into a posture label. Every posture label the
  neon bundle used to compute here read the same eight-integer set the
  driver already carries; adding a table on this side would be a second
  source of truth for a driver-side domain that will not stay in sync
  without a wire-level test - which the driver's tests are the right
  home for.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import HANDSHAKE_FSMS, WALK_FSMS


@tool
async def g1_get_state(driver: Any) -> dict[str, Any]:
    """Return the driver's status plus the arm / loco gate membership answers.

    Read-only. Calls :meth:`~strands_robots.drivers.g1.G1Driver.get_status`
    once, then decides membership of the reported ``fsm_id`` against
    :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` (the arm-SDK
    gate) and :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (the
    locomotion gate). The membership answer is the same one
    :func:`~strands_robots.tools.g1.g1_motion_gates.g1_fsm_admits` would
    compute for the given ``fsm_id``; this verb saves the caller a second
    tool call by carrying it alongside the state read.

    Args:
        driver: An object with an ``async get_status`` method returning
            the driver's status envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). The driver may
            be connected or not; ``get_status`` reports which, and every
            field it does not have yet comes back ``None`` rather than
            raising. Typed :class:`~typing.Any` rather than as ``G1Driver``
            to keep this module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note.

    Returns:
        A dict with ``status``, the driver's ``tool_name`` and ``connected``
        flag, its last-observed ``fsm_id`` / ``mode_machine`` /
        ``battery_pct``, the motion-switcher diagnostics
        (``fsm_mode_name`` / ``fsm_refusal`` / ``motion_switcher_open_error``),
        two decided ``admits_arm`` / ``admits_loco`` booleans, and the
        ``handshake_fsms`` / ``walk_fsms`` id sets the answers were
        computed against (sorted, as lists) so a caller can quote them in
        its own voice. An ``fsm_id`` of ``None`` reports both admit
        booleans as ``False`` - the gate cannot open on a read that never
        arrived.
    """
    envelope = await driver.get_status()
    inner: dict[str, Any] = envelope["content"][0]["json"]

    fsm_id = inner.get("fsm_id")
    admits_arm = isinstance(fsm_id, int) and not isinstance(fsm_id, bool) and fsm_id in HANDSHAKE_FSMS
    admits_loco = isinstance(fsm_id, int) and not isinstance(fsm_id, bool) and fsm_id in WALK_FSMS

    return {
        "status": envelope["status"],
        "tool_name": inner.get("tool_name"),
        "connected": inner.get("connected"),
        "connect_error": inner.get("connect_error"),
        "port": inner.get("port"),
        "network_interface": inner.get("network_interface"),
        "fsm_id": fsm_id,
        "mode_machine": inner.get("mode_machine"),
        "battery_pct": inner.get("battery_pct"),
        "fsm_mode_name": inner.get("fsm_mode_name"),
        "fsm_refusal": inner.get("fsm_refusal"),
        "motion_switcher_open_error": inner.get("motion_switcher_open_error"),
        "admits_arm": admits_arm,
        "admits_loco": admits_loco,
        "handshake_fsms": sorted(HANDSHAKE_FSMS),
        "walk_fsms": sorted(WALK_FSMS),
    }
