"""The audio-call API-id lookup tools name the neon-observed dispatch set.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) exposes
its named helpers (``TtsMaker``, ``GetVolume`` / ``SetVolume``,
``LedControl``, ``PlayStream``, ...) as first-class methods, but one
non-named dispatch surfaces only through raw
``_Call(api_id, payload_json)``: the on-robot ASR (speech-to-text)
API. The neon bundle's
``cagataycali/neon-the-g1/tools/g1_audio.py`` reaches it via
``client._Call(1002, json.dumps(...))``. The
:mod:`strands_robots.tools.g1.g1_audio_call_api_ids` module snapshots
that catalogue into a module-level constant and exposes two
agent-facing verbs -
:func:`g1_list_audio_call_api_ids` (list the whole set) and
:func:`g1_audio_call_api_id_admits` (decide one query) - so a caller
can decide the SDK's ``rc=3103`` refusal decidably before a future
call path is attempted. The tests here fix that contract without
pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off
the module's own snapshot rather than restated in the tests, so a
widen or narrow to the constant surfaces here as a shape change
rather than as a diverging table this file would need to manually
update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  ``_Call`` handler table (the whole point of the port is that the
  snapshot lets a headless host answer). A driver-side wrapper for
  any of the ids that lands later will re-validate against the
  SDK's live handler at wire time; testing the snapshot vs the
  live handler is a driver-side test, not a lookup-side one.
* Whether the firmware on this robot has the id enabled. A
  firmware that registers ``1002`` in ``AudioClient.Init()`` but
  has the underlying ASR service disabled returns a non-zero rc
  at wire time; neither this lookup nor the SDK's own admission
  set can decide it ahead of wire time, and the ``description``
  field on the descriptor says so verbatim. The membership tests
  here grade the snapshot only.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_audio_call_api_ids import (
    _AUDIO_CALL_API_MAP,
    _AUDIO_CALL_WRITE_API_IDS,
    _INVALID_API_CODE,
    _RPC_TIMEOUT_CODE,
    g1_audio_call_api_id_admits,
    g1_list_audio_call_api_ids,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_audio_call_api_ids")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_audio_call_api_ids imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_ids() -> None:
    """The snapshot names every audio API id the neon bundle documented.

    The neon bundle's ``g1_audio.py`` fronts exactly one
    ``AudioClient._Call`` dispatch: ``1002`` (on-robot ASR). The
    id is pinned so a caller widening the map on the neon side
    (e.g. a future firmware that exposes a second audio-side
    ``_Call`` id) updates this cell and the constant together.
    """
    assert len(_AUDIO_CALL_API_MAP) == 1
    assert set(_AUDIO_CALL_API_MAP) == {1002}


def test_the_snapshot_flags_no_write_ids_today() -> None:
    """No audio-side ``_Call`` write is observed today.

    The neon bundle's only observed audio ``_Call`` (``1002``
    ASR) is a read (transcript out). The write frozenset is
    empty; a caller filtering for gate-relevant ids on the audio
    side sees an empty list, and a future firmware id that fronts
    a write will land here and in the ``kind`` field together.
    """
    assert _AUDIO_CALL_WRITE_API_IDS == frozenset()
    for api_id, entry in _AUDIO_CALL_API_MAP.items():
        if api_id in _AUDIO_CALL_WRITE_API_IDS:
            assert entry["kind"] == "write", (
                f"api_id {api_id} is in the write frozenset but its kind field is not 'write'; the two must agree."
            )


def test_every_descriptor_carries_the_required_fields() -> None:
    """Every entry in the snapshot names ``role`` / ``kind`` / ``payload`` / ``description``.

    A widen of the descriptor shape (a new field for a new SDK
    release) lands in one place; a drift where one entry stops
    naming a required field surfaces here rather than at every
    call site.
    """
    required = {"role", "kind", "payload", "description"}
    for api_id, entry in _AUDIO_CALL_API_MAP.items():
        missing = required - set(entry)
        assert missing == set(), (
            f"api_id {api_id} descriptor is missing fields {missing}. Every entry must carry the same shape."
        )
        assert entry["kind"] in {"read", "write"}, f"api_id {api_id} kind {entry['kind']!r} is not 'read' or 'write'."


def test_g1_list_audio_call_api_ids_returns_the_whole_table() -> None:
    """The verb's payload names the map, the ids, and the SDK refusals.

    ``count`` is the size of the module's own snapshot,
    ``audio_call_api_ids`` is one descriptor per admitted id
    (sorted ascending), ``api_ids`` is the sorted id list alone,
    ``write_api_ids`` names the gate-relevant subset (empty
    today), and ``refusals`` names the two refusal codes
    (``3103`` invalid API id, ``3104`` RPC future in flight) with
    the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_audio_call_api_ids)
    assert result["status"] == "success"
    assert result["count"] == len(_AUDIO_CALL_API_MAP)
    assert result["api_ids"] == sorted(_AUDIO_CALL_API_MAP)
    assert result["write_api_ids"] == sorted(_AUDIO_CALL_WRITE_API_IDS)
    assert len(result["audio_call_api_ids"]) == len(_AUDIO_CALL_API_MAP)
    # Every descriptor carries the same field set and reads its flags
    # from the module's constants (not restated in the test body).
    for descriptor in result["audio_call_api_ids"]:
        api_id = descriptor["api_id"]
        entry = _AUDIO_CALL_API_MAP[api_id]
        assert descriptor["role"] == entry["role"]
        assert descriptor["kind"] == entry["kind"]
        assert descriptor["payload"] == entry["payload"]
        assert descriptor["description"] == entry["description"]
        assert descriptor["admits_audio_writes"] is (api_id in _AUDIO_CALL_WRITE_API_IDS)
    codes = {r["code"] for r in result["refusals"]}
    assert codes == {_INVALID_API_CODE, _RPC_TIMEOUT_CODE}
    for refusal in result["refusals"]:
        assert refusal["text"] == ERR_CODES[refusal["code"]]


def test_g1_list_audio_call_api_ids_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh lists and dicts; a mutation on the
    returned ``api_ids`` list or ``audio_call_api_ids``
    descriptors does not leak back into the module's constants.
    This cell is where a share-a-reference regression would
    surface once, not scattered across every call site.
    """
    result = _call(g1_list_audio_call_api_ids)
    result["api_ids"].append(9999)
    result["audio_call_api_ids"][0]["synthetic"] = True
    fresh = _call(g1_list_audio_call_api_ids)
    assert 9999 not in fresh["api_ids"]
    assert "synthetic" not in fresh["audio_call_api_ids"][0]


def test_g1_audio_call_api_id_admits_resolves_the_asr_id() -> None:
    """The observed id ``1002`` is admitted and the descriptor lands.

    ``1002`` is the on-robot ASR ``_Call``; the verb reports
    ``status=success`` and carries the resolved descriptor
    (``api_id``, ``role``, ``kind``, ``payload``, ``description``,
    ``admits_audio_writes``) a future call verb would use to
    decide the follow-up read path. The id is admitted with
    ``admits_audio_writes=False`` (no audio-side ``_Call`` write is
    observed today).
    """
    result = _call(g1_audio_call_api_id_admits, api_id=1002)
    assert result["status"] == "success"
    assert result["api"]["api_id"] == 1002
    assert result["api"]["role"] == "asr"
    assert result["api"]["kind"] == "read"
    assert result["api"]["admits_audio_writes"] is False
    assert "refusal_code" not in result


def test_g1_audio_call_api_id_admits_refuses_an_unknown_id() -> None:
    """An id outside the snapshot is refused with the SDK's ``3103`` code.

    ``9999`` is not in the neon bundle's observed set; a future
    driver-side wrapper's ``_Call`` on it would return
    ``rc=3103`` (\"RPC_CLIENT_API_NOT_REG\"). The verb surfaces
    the same code and its decoded text so a caller planning the
    call sees the number that would land at wire time. This test
    also confirms the loco-side id ``7001`` is refused on the
    audio side: the two clients maintain separate ``_Call``
    admission tables.
    """
    for value in (9999, 7001):
        result = _call(g1_audio_call_api_id_admits, api_id=value)
        assert result["status"] == "error"
        assert result["refusal_code"] == _INVALID_API_CODE
        assert result["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
        assert str(value) in result["reason"]


def test_g1_audio_call_api_id_admits_refuses_a_missing_argument() -> None:
    """A caller reaching the verb with no id is refused decidably.

    A missing argument (``None``) is not resolved through Python's
    coercions; the verb reads it as \"the caller has no id yet\"
    rather than silently returning the whole table. The refusal
    surfaces the ``3103`` code and names the admitted set so the
    caller can pick.
    """
    result = _call(g1_audio_call_api_id_admits)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_API_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
    assert "required" in result["reason"]


def test_g1_audio_call_api_id_admits_refuses_a_bool_argument() -> None:
    """``bool`` is refused because ``int(True)`` would otherwise mis-match.

    Python treats ``True`` as ``int(1)`` and ``False`` as
    ``int(0)``; a caller passing a boolean is a shape mistake,
    not a valid dispatch query. The verb refuses both bool values
    with the ``3103`` code and names the type in the reason.
    """
    for value in (True, False):
        result = _call(g1_audio_call_api_id_admits, api_id=value)
        assert result["status"] == "error", f"api_id={value!r} (bool) should have been refused, got {result!r}."
        assert result["refusal_code"] == _INVALID_API_CODE
        assert "bool" in result["reason"]


def test_g1_audio_call_api_id_admits_refuses_a_non_int_argument() -> None:
    """Non-int arguments (str, float, list) are refused with ``3103``.

    A caller writing ``api_id=\"1002\"`` or ``api_id=1002.0`` is
    making a shape mistake, not naming the SDK's integer id. The
    verb refuses each and names the type in the reason string so
    the caller can correct the call.
    """
    for value in ("1002", 1002.0, [1002], (1002,), 7.5):
        result = _call(g1_audio_call_api_id_admits, api_id=value)
        assert result["status"] == "error", (
            f"api_id={value!r} ({type(value).__name__}) should have been refused, got {result!r}."
        )
        assert result["refusal_code"] == _INVALID_API_CODE
        assert type(value).__name__ in result["reason"] or "not an int" in result["reason"]


def test_the_refusal_codes_all_resolve_through_ERR_CODES() -> None:
    """The two refusal codes the module names are both in ``ERR_CODES``.

    ``3103`` (invalid API) and ``3104`` (RPC future in flight)
    must both round-trip through
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so the
    verb's returned text matches what a driver-side refusal would
    log. A missing entry here would surface as a KeyError; this
    cell fails the shape drift explicitly.
    """
    for code in (_INVALID_API_CODE, _RPC_TIMEOUT_CODE):
        assert code in ERR_CODES, (
            f"refusal code {code} is not in ERR_CODES; the lookup and the "
            "driver's error path must both quote the same table."
        )
