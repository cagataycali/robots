"""Agent-facing lookup for the ``AudioClient._Call`` API ids the neon bundle admits.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) exposes
its named helpers (``TtsMaker``, ``GetVolume`` / ``SetVolume``,
``LedControl``, ``PlayStream``, ...) as first-class methods, but one
non-named dispatch surfaces only through raw
``_Call(api_id, payload_json)``: the on-robot ASR (speech-to-text)
API. The neon bundle's
``cagataycali/neon-the-g1/tools/g1_audio.py`` reaches it via
``client._Call(1002, json.dumps({"duration": ..., "pcm_file": ...}))``
because the SDK registers the id in ``AudioClient.Init()`` but does
not expose a Python helper for it. This module snapshots that single
observed API id and its neon-observed role into a module-level
constant, and exposes two agent-facing verbs -
:func:`g1_list_audio_call_api_ids` (name the whole envelope) and
:func:`g1_audio_call_api_id_admits` (decide one query) - so a caller
can decide the SDK's ``rc=3103`` refusal decidably before a future
driver-side wrapper for the ``_Call`` path is attempted, rather than
triggering the SDK's ``rc=3103`` (\"RPC_CLIENT_API_NOT_REG\") refusal
at wire time. Refs strands-labs/robots#358.

Twin of :mod:`~strands_robots.tools.g1.g1_loco_call_api_ids` (the
loco-side ``_Call`` enumeration verb pair, refs
strands-labs/robots#2992): same envelope shape, same refusal-code
set, same import-hygiene contract. The two modules stay separate
because the loco and audio SDKs are two different singleton clients
- ``LocoClient`` vs ``AudioClient`` - and a caller planning a call
against one side does not want the other side's ids in the returned
envelope.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_asr`` verb wrapped
  ``AudioClient._Call(1002, payload)`` and returned whatever payload
  the firmware reported; the underlying SDK future is a single
  in-flight slot per ``AudioClient`` instance, so concurrent
  ``_Call`` from different threads returns ``rc=3104``
  (\"RPC_CLIENT_API_TIMEOUT\"). Those calls are the same audio RPC
  channel today's :class:`~strands_robots.drivers.g1.G1Driver` does
  not front (the driver's :meth:`~strands_robots.drivers.g1.G1Driver.stream`
  spec declares only ``sensors``/``status``/``stop`` verbs), and a
  future audio-side driver method that fronts the read will land
  alongside the DDS-side speaker output (``rt/audio_msg``) that
  neon's ``G1SpeakerWriter`` uses today. This module ports the
  read-only enumeration half without also introducing a second
  audio writer path the driver does not yet own.
* An SDK re-import. The API-id table is captured here as a
  module-level constant snapshot of the one id the neon bundle
  observed against the real robot; the constant lives here rather
  than being re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_audio_call_api_ids`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358. An SDK release that widens the
  audio-``_Call`` vocabulary (a new voice-side RPC id) is a
  driver-side update; when the driver's audio read method lands,
  its refusal will quote the same ``rc=3103``
  \"RPC_CLIENT_API_NOT_REG\" entry the
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the singleton ``AudioClient`` is currently wedged. The
  neon bundle's ``_loco_call`` treated ``rc=3104``
  (``RPC_CLIENT_API_TIMEOUT``) as a one-shot recovery signal
  (recreate the cached client, retry once, then surface the rc);
  the neon audio path used the SDK's ``_Call`` directly without a
  wrapper, so no matching recovery lock ships on that side. That
  is a driver-instance read (whether the RPC future is in-flight),
  not a snapshot answer; a caller reaching a future driver-side
  wrapper of any of these ids would see the recovery path in the
  returned envelope, and this lookup only names the code the
  recovery path pivots on so a caller planning the call reads the
  same number the write path will surface.
* Whether the firmware on this robot has the id enabled. The neon
  bundle's ``g1_asr`` docstring notes that the on-robot ASR (API
  1002) \"may not be enabled on this firmware\" - a firmware that
  registers the id in ``AudioClient.Init()`` but has the underlying
  ASR service disabled returns ``rc != 0``, and neither this
  lookup nor the SDK's own admission set can decide it ahead of
  wire time. A caller reads the ``description`` field this verb
  returns and knows the id is firmware-gated in ways this snapshot
  does not surface.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the ``AudioClient._Call`` API ids the neon bundle
#: (``cagataycali/neon-the-g1/tools/g1_audio.py``) admits as
#: audio-side RPC dispatches today. Each descriptor names:
#:
#: * ``role`` - the neon helper the id fronts (``asr`` today);
#: * ``kind`` - ``\"read\"`` (the ASR path returns a decoded
#:   transcript payload; there is no known audio-side ``_Call``
#:   write today) or ``\"write\"`` (reserved for a future firmware
#:   that adds one);
#: * ``payload`` - a short prose description of the payload shape
#:   the neon bundle's wrapper constructs (a JSON dict carrying
#:   ``duration`` and optional ``pcm_file`` for ``1002``).
#:
#: The label table lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the audio ``_Call``-side of the conversation; a
#: caller that needs the loco-side ``_Call`` reaches
#: :mod:`~strands_robots.tools.g1.g1_loco_call_api_ids` directly.
#: Colocating the id table with the enumeration verb mirrors
#: ``_LOCO_CALL_API_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_loco_call_api_ids` and
#: ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_AUDIO_CALL_API_MAP: dict[int, dict[str, str]] = {
    1002: {
        "role": "asr",
        "kind": "read",
        "payload": '{"duration": <float>, "pcm_file": <path?>}',
        "description": (
            "On-robot ASR (automatic speech recognition). The neon "
            "bundle's ``g1_asr`` helper fronts this by JSON-encoding "
            "a payload with ``duration`` (seconds to listen for on "
            "the robot's own mic) and an optional ``pcm_file`` "
            "(path on the robot to a raw PCM file to transcribe). "
            "The SDK registers the id in ``AudioClient.Init()`` but "
            "does not expose a Python helper; the exact response "
            "schema is undocumented (the neon wrapper returns the "
            "raw response for the caller to inspect). Firmware-gated: "
            "a build with ASR disabled returns a non-zero rc even "
            "though the id is admitted in the SDK table."
        ),
    },
}

#: The subset of :data:`_AUDIO_CALL_API_MAP` that fronts an
#: audio-shaped *write* to the SDK. Empty today; the only observed
#: id (``1002`` ASR) is a read. Called out separately for shape
#: parity with :data:`~strands_robots.tools.g1.g1_loco_call_api_ids._LOCO_CALL_WRITE_API_IDS`
#: so a caller filtering for writes on either side (loco or audio)
#: reads the same field on both.
_AUDIO_CALL_WRITE_API_IDS: frozenset[int] = frozenset()

#: The error-table entry the SDK's ``_Call`` returns for an API id
#: outside :data:`_AUDIO_CALL_API_MAP` (a mis-typed number, or an id
#: from a firmware release the SDK on this host does not know
#: about). Named here so the returned envelope carries the exact
#: refusal string a future driver-side wrapper would surface, and
#: so a re-wording of it lands in one place instead of drifting
#: between the SDK-side log and this lookup.
_INVALID_API_CODE: int = 3103

#: The error-table entry the SDK returns when the AudioClient's
#: RPC future is already in flight (\"RPC_CLIENT_API_TIMEOUT\").
#: The neon bundle did not wrap the audio ``_Call`` under a
#: dedicated single-writer lock (the way it did for
#: :data:`~strands_robots.tools.g1.g1_loco_call_api_ids._RPC_TIMEOUT_CODE`
#: on the loco side), but the same singleton-client shape applies:
#: two threads reaching ``AudioClient._Call`` at once will collide
#: on the RPC future. Named here because
#: :func:`g1_list_audio_call_api_ids` surfaces it alongside the
#: ``3103`` on the returned refusal list so a caller sees both the
#: shape refusal (bad id) and the concurrency refusal (client
#: wedged) at once.
_RPC_TIMEOUT_CODE: int = 3104


def _describe(api_id: int) -> dict[str, Any]:
    """Build the per-id descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_audio_call_api_ids` so
    :func:`g1_audio_call_api_id_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is
    touched.
    """
    entry = _AUDIO_CALL_API_MAP[api_id]
    return {
        "api_id": api_id,
        "role": entry["role"],
        "kind": entry["kind"],
        "payload": entry["payload"],
        "description": entry["description"],
        "admits_audio_writes": api_id in _AUDIO_CALL_WRITE_API_IDS,
    }


@tool
def g1_list_audio_call_api_ids() -> dict[str, Any]:
    """Return the ``AudioClient._Call`` API ids the neon bundle admits.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side
    wrapper for any of these ids is called, so a caller can compare
    an intended API id against the set the neon bundle observed
    against the real robot, and decide alongside that whether the
    id is a read (safe, though possibly firmware-gated) or a write
    (none observed today; the field is surfaced for parity with
    :func:`~strands_robots.tools.g1.g1_loco_call_api_ids.g1_list_loco_call_api_ids`).

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        admitted API ids; an ``audio_call_api_ids`` list of
        descriptors (one per admitted id, sorted ascending)
        carrying ``api_id``, ``role`` (the neon helper name),
        ``kind`` (``\"read\"`` or ``\"write\"``), ``payload`` (the
        payload shape the neon bundle constructs), ``description``
        (the neon-observed purpose and any firmware-gating notes),
        and ``admits_audio_writes`` (``False`` on every observed
        id today; surfaced for shape parity with
        :mod:`~strands_robots.tools.g1.g1_loco_call_api_ids`); an
        ``api_ids`` list of just the ids in sorted order; a
        ``write_api_ids`` list mirroring
        :data:`_AUDIO_CALL_WRITE_API_IDS` (empty today); and a
        ``refusals`` list carrying the two refusal codes (``3103``
        invalid API id, ``3104`` RPC future in flight) and their
        decoded text that a future call verb would surface. Every
        field is a snapshot of an SDK or neon constant; no dynamic
        decode runs here.
    """
    api_ids = sorted(_AUDIO_CALL_API_MAP)
    return {
        "status": "success",
        "count": len(_AUDIO_CALL_API_MAP),
        "audio_call_api_ids": [_describe(api_id) for api_id in api_ids],
        "api_ids": api_ids,
        "write_api_ids": sorted(_AUDIO_CALL_WRITE_API_IDS),
        "refusals": [
            {"code": _INVALID_API_CODE, "text": ERR_CODES[_INVALID_API_CODE]},
            {"code": _RPC_TIMEOUT_CODE, "text": ERR_CODES[_RPC_TIMEOUT_CODE]},
        ],
    }


@tool
def g1_audio_call_api_id_admits(api_id: int | None = None) -> dict[str, Any]:
    """Decide whether ``api_id`` is inside the neon-observed audio dispatch set.

    Read-only. Compares one argument against the neon-observed
    :data:`_AUDIO_CALL_API_MAP` and reports the admitted descriptor
    on match, or the ``3103`` refusal code a future driver-side
    wrapper would quote on miss. No driver instance, no DDS, no
    SDK: the decision reads only module-level constants and the
    argument itself.

    An id inside the admitted set is *not* the same as an admitted
    call: any id is refused with ``rc=3104`` while the singleton
    ``AudioClient``'s RPC future is in flight, and a firmware that
    registers the id but has the underlying service disabled
    returns a non-zero rc at wire time. Neither is a snapshot
    answer; both are live-driver reads a caller reaches after this
    verb admits the id. The returned payload's ``kind`` field
    names ``\"read\"`` or ``\"write\"`` so a caller comparing an
    intended call against both conditions (membership + firmware
    availability) sees which side of the gate the id lands on.

    Args:
        api_id: The API id to check. Must be an ``int``; ``bool``
            is refused with the ``3103`` code because ``int(True)``
            is ``1`` and a passed-through boolean is a caller
            mistake, not a valid dispatch query. A missing argument
            (``None``) is refused decidably rather than treated as
            a default.

    Returns:
        A dict with ``status``; on admit, an ``api`` descriptor
        with ``api_id``, ``role``, ``kind``, ``payload``,
        ``description``, and ``admits_audio_writes`` (the same
        shape :func:`g1_list_audio_call_api_ids` returns). On
        refuse, ``refusal_code`` and ``refusal_text`` name the
        ``3103`` code and its decoded text, plus a ``reason``
        string that names why the argument was refused (missing
        argument, bool argument, non-int argument, or unknown API
        id).
    """
    if api_id is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id is required; pass one of {sorted(_AUDIO_CALL_API_MAP)} "
                "so the lookup is decidable. Refs strands-labs/robots#358."
            ),
        }
    if isinstance(api_id, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is a bool; pass one of "
                f"{sorted(_AUDIO_CALL_API_MAP)} as an int. "
                "Refs strands-labs/robots#358."
            ),
        }
    if not isinstance(api_id, int):
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is not an int; pass one of "
                f"{sorted(_AUDIO_CALL_API_MAP)} as an int. "
                "Refs strands-labs/robots#358."
            ),
        }
    if api_id not in _AUDIO_CALL_API_MAP:
        return {
            "status": "error",
            "refusal_code": _INVALID_API_CODE,
            "refusal_text": ERR_CODES[_INVALID_API_CODE],
            "reason": (
                f"api_id={api_id!r} is not in the admitted set "
                f"{sorted(_AUDIO_CALL_API_MAP)}. "
                "Refs strands-labs/robots#358."
            ),
        }
    return {
        "status": "success",
        "api": _describe(api_id),
    }
