"""The loco-call API-id lookup tools name the neon-observed dispatch set.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
handful of read-side motion-state queries and one write-side setter
as raw ``_Call(api_id, payload_json)`` dispatches. The neon bundle's
``cagataycali/neon-the-g1/tools/_g1_common.py`` catalogues those ids
across its ``read_fsm_id`` / ``read_fsm_mode`` / ``read_balance_mode``
/ ``read_swing_height`` / ``read_stand_height`` / ``set_swing_height``
helpers. The :mod:`strands_robots.tools.g1.g1_loco_call_api_ids`
module snapshots that catalogue into a module-level constant and
exposes two agent-facing verbs -
:func:`g1_list_loco_call_api_ids` (list the whole set) and
:func:`g1_loco_call_api_id_admits` (decide one query) - so a caller
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
* Which FSM ids the locomotion write gate admits on. The verb
  surfaces :data:`WALK_FSMS` verbatim because a caller planning a
  write-side ``_Call`` compares the target against the write gate
  too; the membership rule for that gate is already pinned in
  :mod:`tests.drivers.test_g1_motion_gates_reads_the_driver_contract`,
  so this file only checks that the surfaced set matches what the
  driver's constant ships.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_loco_call_api_ids import (
    _GATE_REFUSAL_CODE,
    _INVALID_API_CODE,
    _LOCO_CALL_API_MAP,
    _LOCO_CALL_WRITE_API_IDS,
    _RPC_TIMEOUT_CODE,
    g1_list_loco_call_api_ids,
    g1_loco_call_api_id_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim. This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up (refs strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_loco_call_api_ids")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_loco_call_api_ids imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_ids() -> None:
    """The snapshot names every API id the neon bundle documented.

    The neon bundle's ``_g1_common.py`` fronts six ``_Call``
    dispatches: ``7001`` (fsm id), ``7002`` (fsm mode), ``7003``
    (balance mode), ``7004`` (swing height read), ``7005`` (stand
    height read) and ``7103`` (swing height write). The count is
    pinned rather than listed name-by-name so a caller widening the
    map on the neon side updates one number here rather than six
    assertions.
    """
    assert len(_LOCO_CALL_API_MAP) == 6
    assert set(_LOCO_CALL_API_MAP) == {7001, 7002, 7003, 7004, 7005, 7103}


def test_the_snapshot_flags_the_write_ids() -> None:
    """The one write-shaped id is ``7103`` (swing height setter).

    The neon bundle's ``set_swing_height`` helper is the only write
    fronted through ``_Call``; every other id in the map is a read.
    This cell pins the write set so a caller filtering for
    gate-relevant ids compares against the frozenset directly, and
    so a widen of the write set (e.g. a future stand-height setter)
    lands here first rather than as a divergence between the ``kind``
    field and the write frozenset.
    """
    assert _LOCO_CALL_WRITE_API_IDS == frozenset({7103})
    for api_id in _LOCO_CALL_WRITE_API_IDS:
        assert api_id in _LOCO_CALL_API_MAP, (
            f"write id {api_id} is flagged but not in the API-id snapshot; the flag can only apply to admitted ids."
        )
        assert _LOCO_CALL_API_MAP[api_id]["kind"] == "write", (
            f"api_id {api_id} is in the write frozenset but its kind field is not 'write'; the two must agree."
        )


def test_every_descriptor_carries_the_required_fields() -> None:
    """Every entry in the snapshot names ``role`` / ``kind`` / ``payload``.

    A widen of the descriptor shape (a new field for a new SDK
    release) lands in one place; a drift where one entry stops
    naming a required field surfaces here rather than at every call
    site.
    """
    required = {"role", "kind", "payload", "description"}
    for api_id, entry in _LOCO_CALL_API_MAP.items():
        missing = required - set(entry)
        assert missing == set(), (
            f"api_id {api_id} descriptor is missing fields {missing}. Every entry must carry the same shape."
        )
        assert entry["kind"] in {"read", "write"}, f"api_id {api_id} kind {entry['kind']!r} is not 'read' or 'write'."


def test_g1_list_loco_call_api_ids_returns_the_whole_table() -> None:
    """The verb's payload names the map, the ids, and the SDK refusals.

    ``count`` is the size of the module's own snapshot,
    ``loco_call_api_ids`` is one descriptor per admitted id (sorted
    ascending), ``api_ids`` is the sorted id list alone,
    ``write_api_ids`` names the gate-relevant subset,
    ``loco_ready_fsm_ids`` mirrors the driver's write-gate set, and
    ``refusals`` names the three refusal codes (``3103`` invalid API
    id, ``3104`` RPC future in flight, ``7404`` gate-refused write)
    with the decoded text
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.
    """
    result = _call(g1_list_loco_call_api_ids)
    assert result["status"] == "success"
    assert result["count"] == len(_LOCO_CALL_API_MAP)
    assert result["api_ids"] == sorted(_LOCO_CALL_API_MAP)
    assert result["write_api_ids"] == sorted(_LOCO_CALL_WRITE_API_IDS)
    assert result["loco_ready_fsm_ids"] == sorted(WALK_FSMS)
    assert len(result["loco_call_api_ids"]) == len(_LOCO_CALL_API_MAP)
    # Every descriptor carries the same field set and reads its flags
    # from the module's constants (not restated in the test body).
    for descriptor in result["loco_call_api_ids"]:
        api_id = descriptor["api_id"]
        entry = _LOCO_CALL_API_MAP[api_id]
        assert descriptor["role"] == entry["role"]
        assert descriptor["kind"] == entry["kind"]
        assert descriptor["payload"] == entry["payload"]
        assert descriptor["description"] == entry["description"]
        assert descriptor["admits_loco_writes"] is (api_id in _LOCO_CALL_WRITE_API_IDS)
    codes = {r["code"] for r in result["refusals"]}
    assert codes == {_INVALID_API_CODE, _RPC_TIMEOUT_CODE, _GATE_REFUSAL_CODE}
    for refusal in result["refusals"]:
        assert refusal["text"] == ERR_CODES[refusal["code"]]


def test_g1_list_loco_call_api_ids_returns_fresh_containers() -> None:
    """A caller mutating the payload cannot poison the module snapshot.

    The verb returns fresh lists and dicts; a mutation on the
    returned ``api_ids`` list or ``loco_call_api_ids`` descriptors
    does not leak back into the module's constants. This cell is
    where a share-a-reference regression would surface once, not
    scattered across every call site.
    """
    result = _call(g1_list_loco_call_api_ids)
    result["api_ids"].append(9999)
    result["loco_call_api_ids"][0]["synthetic"] = True
    fresh = _call(g1_list_loco_call_api_ids)
    assert 9999 not in fresh["api_ids"]
    assert "synthetic" not in fresh["loco_call_api_ids"][0]


def test_g1_loco_call_api_id_admits_resolves_a_read_id() -> None:
    """An id inside the neon set is admitted and the descriptor lands.

    ``7001`` is ``read_fsm_id``; the verb reports ``status=success``
    and carries the resolved descriptor (``api_id``, ``role``,
    ``kind``, ``payload``, ``description``, ``admits_loco_writes``)
    a future call verb would use to decide the follow-up read path.
    A read id is admitted with ``admits_loco_writes=False`` (the
    driver's write gate does not fire on reads).
    """
    result = _call(g1_loco_call_api_id_admits, api_id=7001)
    assert result["status"] == "success"
    assert result["api"]["api_id"] == 7001
    assert result["api"]["role"] == "read_fsm_id"
    assert result["api"]["kind"] == "read"
    assert result["api"]["admits_loco_writes"] is False
    assert "refusal_code" not in result


def test_g1_loco_call_api_id_admits_resolves_the_write_id() -> None:
    """The write id ``7103`` is admitted with ``admits_loco_writes=True``.

    ``7103`` is ``set_swing_height``; the verb reports
    ``status=success`` and the descriptor flags the id as a
    gate-relevant write (a caller planning the call also needs an
    fsm inside :data:`WALK_FSMS`).
    """
    result = _call(g1_loco_call_api_id_admits, api_id=7103)
    assert result["status"] == "success"
    assert result["api"]["api_id"] == 7103
    assert result["api"]["role"] == "set_swing_height"
    assert result["api"]["kind"] == "write"
    assert result["api"]["admits_loco_writes"] is True
    assert result["api"]["payload"] == '{"data": <float>}'


def test_g1_loco_call_api_id_admits_refuses_an_unknown_id() -> None:
    """An id outside the snapshot is refused with the SDK's ``3103`` code.

    ``9999`` is not in the neon bundle's observed set; a future
    driver-side wrapper's ``_Call`` on it would return
    ``rc=3103`` ("RPC_CLIENT_API_NOT_REG"). The verb surfaces the
    same code and its decoded text so a caller planning the write
    sees the number that would land at wire time.
    """
    result = _call(g1_loco_call_api_id_admits, api_id=9999)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_API_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
    assert "9999" in result["reason"]


def test_g1_loco_call_api_id_admits_refuses_a_missing_argument() -> None:
    """A caller reaching the verb with no id is refused decidably.

    A missing argument (``None``) is not resolved through Python's
    coercions; the verb reads it as "the caller has no id yet"
    rather than silently returning the whole table. The refusal
    surfaces the ``3103`` code and names the admitted set so the
    caller can pick.
    """
    result = _call(g1_loco_call_api_id_admits)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_API_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
    assert "required" in result["reason"]


def test_g1_loco_call_api_id_admits_refuses_a_bool_argument() -> None:
    """``bool`` is refused because ``int(True)`` would otherwise mis-match.

    Python treats ``True`` as ``int(1)`` and ``False`` as ``int(0)``;
    a caller passing a boolean is a shape mistake, not a valid
    dispatch query. The verb refuses both bool values with the
    ``3103`` code and names the type in the reason.
    """
    for value in (True, False):
        result = _call(g1_loco_call_api_id_admits, api_id=value)
        assert result["status"] == "error", f"api_id={value!r} (bool) should have been refused, got {result!r}."
        assert result["refusal_code"] == _INVALID_API_CODE
        assert "bool" in result["reason"]


def test_g1_loco_call_api_id_admits_refuses_a_non_int_argument() -> None:
    """Non-int arguments (str, float, list) are refused with ``3103``.

    A caller writing ``api_id="7001"`` or ``api_id=7001.0`` is
    making a shape mistake, not naming the SDK's integer id. The
    verb refuses each and names the type in the reason string so
    the caller can correct the call.
    """
    for value in ("7001", 7001.0, [7001], (7001,), 7.5):
        result = _call(g1_loco_call_api_id_admits, api_id=value)
        assert result["status"] == "error", (
            f"api_id={value!r} ({type(value).__name__}) should have been refused, got {result!r}."
        )
        assert result["refusal_code"] == _INVALID_API_CODE
        assert type(value).__name__ in result["reason"] or "not an int" in result["reason"]


def test_the_refusal_codes_all_resolve_through_ERR_CODES() -> None:
    """The three refusal codes the module names are all in ``ERR_CODES``.

    ``3103`` (invalid API), ``3104`` (RPC future in flight), and
    ``7404`` (gate-refused write) must all round-trip through
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` so the
    verb's returned text matches what a driver-side refusal would
    log. A missing entry here would surface as a KeyError; this
    cell fails the shape drift explicitly.
    """
    for code in (_INVALID_API_CODE, _RPC_TIMEOUT_CODE, _GATE_REFUSAL_CODE):
        assert code in ERR_CODES, (
            f"refusal code {code} is not in ERR_CODES; the lookup and the "
            "driver's error path must both quote the same table."
        )
