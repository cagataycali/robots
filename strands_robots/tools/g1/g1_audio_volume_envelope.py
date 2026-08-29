"""Agent-facing lookup for the volume envelope ``AudioClient.SetVolume`` admits.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) exposes
``SetVolume(volume)`` and its ``GetVolume()`` companion. The SDK itself
places *no* clamps on the ``volume`` argument: a caller that passes
``volume=250`` reaches the audio controller unchanged, and the
controller's own behaviour above the neon-bundle-observed usable range
is undefined - the G1 has no runaway guard on that write path. The neon
bundle's field notes on the SDK
(``cagataycali/neon-the-g1/tools/use_unitree.py`` names the range as
``SetVolume(volume) / GetVolume() → 0-100``, matching Unitree's own
service documentation for the audio API), so this module surfaces the
observed integer range as module-level constants and exposes two
agent-facing verbs so a caller can decide the refusal decidably before
a future driver-side wrapper for ``SetVolume`` fires, rather than
pinning the range inside the write path where the refusal is invisible
to the planner.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_set_volume`` verb wrapped
  ``AudioClient.SetVolume`` under a single-writer lock; that write is
  the same audio-service path :class:`AudioClient` publishes on today
  from the driver-side singletons named in
  :mod:`~strands_robots.tools.g1._g1_common`. A future driver method
  that fronts ``SetVolume`` will land the write verb; refs
  strands-labs/robots#358 for the SDK-facing gate work that write
  belongs on. This module ports the read-only envelope half without
  also introducing a second audio writer path the driver does not yet
  own.
* An SDK re-import. The clamp table is captured here as module-level
  constants so ``import strands_robots.tools.g1.g1_audio_volume_envelope``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs strands-labs/robots#358.
  A revision of the observed bounds is a driver-side update; when the
  driver's volume method lands, its refusal will surface the same
  module-local :data:`_REFUSAL_TEXT` this module names for a bounds
  violation.

Why this module does not quote a driver-side ``rc``.

The G1 driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
gates the *motion* surface (arm-SDK writes on ``rt/lowcmd``); its FSM
rejections are the ``7404`` entry in
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
(``"Invalid FSM id - need FSM in {500, 501, 801}"``). ``AudioClient`` is
an RPC service on a separate channel from ``rt/lowcmd``, and the audio
SDK ships no distinct rc for a bounds-violated volume argument.
Borrowing ``7404`` on an audio refusal would hand an agent planner a
motion-FSM remedy (``"need FSM in {500, 501, 801}"``) for a bounds
violation on a value that has nothing to do with the locomotion FSM.
The refusal shape this module returns names the numeric bound
violation in module-local text so a planner reads a remedy that
matches the surface, and a future driver-side ``SetVolume`` wrapper
will surface the same module-local text - not a re-borrowed motion
code.

What this module does not decide.

* The live audio state. Whether the driver's ``AudioClient`` singleton
  is currently constructed, whether a ``PlayStream`` write is in
  flight, whether a ``TtsMaker`` job holds the audio bus: none of
  those live-instance reads run here. A caller planning a
  ``SetVolume`` write reads this verb's ``envelope`` to decide
  whether their value is inside the observed range and reads the
  driver's own liveness signal separately to decide whether the
  write path is currently free.
* Whether ``rt/lowcmd`` is currently held by another writer. The
  audio bus does not run on ``rt/lowcmd`` - ``AudioClient`` is an
  RPC service on a separate channel - but the same locking rule
  applies: the driver's audio singleton reports contention at wire
  time; a caller planning a write cannot decide it without opening
  the channel itself, and this module opens no channel.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The lower clamp on ``volume`` (integer percent). The neon bundle's
#: field notes on the SDK name the observed range as ``0-100`` with
#: ``0`` meaning muted and ``100`` meaning maximum; the SDK's
#: ``SetVolume`` does not itself refuse a negative integer, so this
#: bound is applied at the verb boundary rather than inside the SDK.
#: Named as an inclusive bound (``value < bound`` refuses, ``value ==
#: bound`` admits) because a caller muting the speaker with
#: ``volume=0`` is a legitimate command, not the shape-error the
#: refusal exists to catch.
_VOLUME_MIN: int = 0

#: The upper clamp on ``volume``. The neon bundle's field notes name
#: ``100`` as the maximum; above it the audio controller's clipping
#: behaviour is undefined. Named as an inclusive bound like the
#: lower clamp, so a caller writing ``volume=100`` (full volume) is
#: admitted rather than tripping an off-by-one.
_VOLUME_MAX: int = 100

#: The module-local refusal text every ``g1_volume_admits`` refusal
#: quotes when the caller's ``volume`` argument sits outside the
#: neon-bundle-observed envelope. Named here rather than borrowed
#: from :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
#: because the audio SDK ships no distinct rc for a bounds-violated
#: volume and the motion-FSM ``7404`` entry (its nearest neighbour)
#: reads ``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a
#: remedy that points a planner at locomotion FSM transitions to
#: fix an audio argument. Surfacing the module-local text keeps
#: the refusal payload's remedy on the same surface the write
#: belongs on, and a future driver-side ``SetVolume`` wrapper will
#: surface this same text rather than re-borrowing a motion code.
_REFUSAL_TEXT: str = f"volume out of envelope - need volume in [{_VOLUME_MIN}, {_VOLUME_MAX}]"


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_audio_volume_envelope`
    so :func:`g1_volume_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    return {
        "volume_min": _VOLUME_MIN,
        "volume_max": _VOLUME_MAX,
    }


@tool
def g1_list_audio_volume_envelope() -> dict[str, Any]:
    """Return the volume clamp envelope the neon bundle observed as usable.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for ``AudioClient.SetVolume`` is called, so a caller can compare
    an intended ``volume`` argument against the envelope the neon
    bundle observed as usable, and can carry the module-local
    refusal text a driver-side wrapper would surface on a bounds
    violation.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        every clamp the neon bundle applied (``volume_min``,
        ``volume_max``); and a ``refusals`` list carrying a single
        descriptor with the module-local :data:`_REFUSAL_TEXT` a
        future write verb would surface on a bounds violation.
        Every field is a snapshot of an observed bound or a
        module-local text; no dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_volume_admits(volume: int = 0) -> dict[str, Any]:
    """Decide whether a ``volume`` argument sits inside the envelope.

    Read-only. Compares the argument against the clamps
    :func:`g1_list_audio_volume_envelope` returns and reports the
    refusal shape if any bound is violated. No driver instance, no
    DDS, no SDK: the decision reads only module-level constants and
    the argument itself.

    A ``volume`` inside the envelope is *not* the same as an admitted
    write: the driver's audio singleton may refuse on liveness
    grounds (an in-flight ``PlayStream``, a not-yet-constructed
    ``AudioClient``), which this verb does not read (that is a live
    driver-instance query answered by a future audio state verb).
    The returned envelope names only the numeric bound decision.

    Args:
        volume: integer percent in ``[0, 100]``. Refused below
            ``volume_min`` and above ``volume_max``. Boolean values
            are refused explicitly at the boundary because Python's
            ``bool`` is a subclass of ``int``, so a caller passing
            ``True`` would otherwise silently look up ``1`` (a
            legitimate low volume) and hide the type mistake; naming
            the refusal at the boundary surfaces the mistake instead.
            Non-integer numeric values (``float``, ``Decimal``) are
            refused with the same shape so a caller passing
            ``volume=50.0`` sees an actionable refusal rather than a
            silent truncation the SDK would perform.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value is inside the clamp pair; a ``refusals`` list of
        refusal descriptors, each carrying the dimension name, the
        offending value, the clamp it violated, and the module-local
        :data:`_REFUSAL_TEXT` a driver-side wrapper would surface if
        the write were attempted while the value is outside the
        envelope; the same ``envelope`` sub-dict
        :func:`g1_list_audio_volume_envelope` returns. On an
        admitted value the ``refusals`` list is empty; on a rejected
        value the single violated bound is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(value: Any, bound_key: str, bound: int, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "volume",
                "value": value,
                "bound_key": bound_key,
                "bound": bound,
                "comparison": cmp,
                "text": _REFUSAL_TEXT,
            }
        )

    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 and hide a type mistake at the boundary.
    if isinstance(volume, bool):
        _reject(volume, "volume_min", _VOLUME_MIN, "non-int")
    elif not isinstance(volume, int):
        _reject(volume, "volume_min", _VOLUME_MIN, "non-int")
    else:
        v = int(volume)
        if v < _VOLUME_MIN:
            _reject(volume, "volume_min", _VOLUME_MIN, "value < bound")
        elif v > _VOLUME_MAX:
            _reject(volume, "volume_max", _VOLUME_MAX, "value > bound")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
