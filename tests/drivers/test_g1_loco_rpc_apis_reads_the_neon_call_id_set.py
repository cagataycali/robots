"""The loco-RPC-api lookup tools name what ``LocoClient._Call`` admits.

The Unitree G1 locomotion SDK
(:class:`unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`) exposes a
raw ``_Call(api_id, payload)`` transport under its high-level helpers
(``SetFsmId``, ``BalanceStand``, ``StopMove`` ...); the neon bundle's
``_loco_call`` helper
(``cagataycali/neon-the-g1/tools/_g1_common.py``) reaches directly
through ``_Call`` for the api ids the SDK does not surface as named
Python methods, and stashed the observed mapping in the read helpers
``read_fsm_id`` .. ``read_stand_height`` and the write helper
``set_swing_height``. The :mod:`strands_robots.tools.g1.g1_loco_rpc_apis`
module snapshots that observed api-id set into a module-level dict and
exposes two agent-facing verbs -
:func:`g1_list_loco_rpc_apis` (name the whole set) and
:func:`g1_loco_rpc_api_admits` (decide one query) - so a caller can
decide the refusal decidably before a future raw ``_Call`` path is
attempted. The tests here fix that contract without pulling the SDK:
the module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen
or narrow to the observed set surfaces here as a shape change rather
than as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The snapshot is the neon
  bundle's observed api-id set, not the SDK's own admissions (the
  SDK's ``_Call`` accepts any integer and returns rc=3102 or 3104 on
  a transport failure, silently on an unknown id at the handler
  boundary). A driver-side wrapper for ``_Call`` that lands later
  will re-check the api id at wire time and its refusal string will
  quote the same ``3102`` / ``3104`` / ``7404`` codes this snapshot
  names.
* Whether the driver's live ``fsm_id`` sits inside
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`. That is a
  live driver-instance read and belongs on
  :mod:`~strands_robots.tools.g1.g1_state` /
  :mod:`~strands_robots.tools.g1.g1_motion_gates`; this verb
  surfaces the set as a snapshot so a caller comparing an intended
  write against both conditions has the FSM set on hand.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES, WALK_FSMS
from strands_robots.tools.g1.g1_loco_rpc_apis import (
    _INVALID_API_CODE,
    _LOCO_RPC_APIS,
    _READ_APIS,
    _TRANSPORT_REFUSAL_CODES,
    _WRITE_APIS,
    g1_list_loco_rpc_apis,
    g1_loco_rpc_api_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent (refs strands-labs/robots#358); a module that
    pulled a submodule at import time would break every headless CI
    runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_loco_rpc_apis")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_loco_rpc_apis imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the SDK "
        "loads only from function bodies (refs strands-labs/robots#358)."
    )


def test_the_read_and_write_partitions_cover_every_admitted_api() -> None:
    """Every api id in :data:`_LOCO_RPC_APIS` is either read or write.

    The verb's ``kind`` field partitions the admitted set into two
    labels; a caller reading a descriptor sees exactly one of them.
    A widen that added a third label would surface here as a
    disjoint-partition assertion failure, so the shape change lands
    in review rather than as a silently-broken descriptor at the
    call site.
    """
    read_only = _READ_APIS - _WRITE_APIS
    write_only = _WRITE_APIS - _READ_APIS
    both = _READ_APIS & _WRITE_APIS
    assert both == frozenset(), (
        f"api ids overlap read and write partitions: {sorted(both)}. "
        "Every admitted api id must sit in exactly one of "
        "_READ_APIS or _WRITE_APIS."
    )
    covered = read_only | write_only
    admitted = frozenset(_LOCO_RPC_APIS)
    assert covered == admitted, (
        f"partition mismatch: _READ_APIS | _WRITE_APIS = {sorted(covered)}, "
        f"_LOCO_RPC_APIS = {sorted(admitted)}. Every admitted api id must "
        "sit in exactly one of the two partitions."
    )


def test_the_transport_refusal_codes_are_named_in_the_shared_table() -> None:
    """The transport refusal codes come from :data:`ERR_CODES`.

    Both ``3102`` and ``3104`` are transport-level rc codes the neon
    bundle documented as reachable from ``_Call``; they are the codes
    a future driver-side wrapper for ``_Call`` would surface, so the
    lookup here quotes them from the same table
    :mod:`~strands_robots.tools.g1._g1_common` ships. A rewording of
    either entry there surfaces here as a text drift the shared
    table test would also catch.
    """
    for code in _TRANSPORT_REFUSAL_CODES:
        assert code in ERR_CODES, (
            f"transport refusal code {code} is not in ERR_CODES; the "
            "lookup would quote a code the shared table does not name."
        )


def test_the_invalid_api_code_is_the_shared_gate_refusal_code() -> None:
    """The invalid-api-id code is the same ``7404`` gate-refusal code.

    The SDK does not ship a dedicated "invalid loco api id" code; the
    neon bundle's helpers refuse at the boundary rather than reach
    the handler, so the lookup here quotes the same ``7404``
    gate-refusal shape a locomotion-write refusal uses. A future SDK
    release that adds a distinct code lands as a shape change here
    without also renaming the write-refusal code
    :mod:`~strands_robots.tools.g1.g1_balance_modes` quotes.
    """
    assert _INVALID_API_CODE == 7404, (
        f"invalid-api-id code is {_INVALID_API_CODE}, expected 7404. "
        "The lookup quotes the same gate-refusal code the driver's "
        "_check_motion_gates uses so both sides refuse with the same "
        "text (refs strands-labs/robots#2916)."
    )
    assert _INVALID_API_CODE in ERR_CODES, (
        f"invalid-api-id code {_INVALID_API_CODE} is not in ERR_CODES; "
        "the lookup would quote a code the shared table does not name."
    )


def test_the_list_verb_returns_every_admitted_api_sorted() -> None:
    """The list verb names every api id in :data:`_LOCO_RPC_APIS`.

    The verb surfaces the module's own snapshot, so a widen or
    narrow to the observed set surfaces here as a count mismatch
    rather than as a diverging manual table this test would need to
    update. Sorted by ``api_id`` ascending so the descriptor order
    is stable across reads.
    """
    payload = _call(g1_list_loco_rpc_apis)
    assert payload["status"] == "success"
    apis = payload["apis"]
    admitted_ids = sorted(_LOCO_RPC_APIS)
    seen_ids = [entry["api_id"] for entry in apis]
    assert seen_ids == admitted_ids, (
        f"g1_list_loco_rpc_apis returned api ids {seen_ids}, expected the sorted admitted set {admitted_ids}."
    )


def test_the_list_verb_describes_every_api_with_the_expected_fields() -> None:
    """Each api descriptor carries the four shared fields.

    The four fields are the same shape
    :func:`g1_loco_rpc_api_admits`'s admitted-path payload returns,
    so a caller comparing the two verbs sees identical descriptors.
    A widen to the descriptor lands in
    :func:`~strands_robots.tools.g1.g1_loco_rpc_apis._describe` and
    surfaces here as a shape drift.
    """
    payload = _call(g1_list_loco_rpc_apis)
    for entry in payload["apis"]:
        assert set(entry.keys()) == {
            "api_id",
            "operation",
            "kind",
            "touches_motion_gate",
        }, f"descriptor {entry} does not match the shared shape (api_id, operation, kind, touches_motion_gate)."
        assert entry["kind"] in {"read", "write"}, (
            f"descriptor {entry} names an unknown kind; the partition test would also catch this on the constants side."
        )
        assert entry["touches_motion_gate"] == (entry["api_id"] in _WRITE_APIS), (
            f"descriptor {entry} claims touches_motion_gate={entry['touches_motion_gate']} "
            f"but api_id={entry['api_id']} sits in _WRITE_APIS="
            f"{entry['api_id'] in _WRITE_APIS}. The write partition is the "
            "one condition the motion gate refuses on."
        )


def test_the_list_verb_names_the_walk_ready_fsm_set() -> None:
    """The list verb quotes :data:`WALK_FSMS` for the gate-set half.

    A caller planning a write-side api call (``7103`` today) reads
    both the api-id descriptor and the walk-ready set to decide
    whether the driver's motion gate would admit the write; the
    verb ships both so the two membership tests can happen on the
    caller side without a second driver-side query.
    """
    payload = _call(g1_list_loco_rpc_apis)
    assert payload["walk_ready_fsm_ids"] == sorted(WALK_FSMS), (
        f"g1_list_loco_rpc_apis returned walk_ready_fsm_ids="
        f"{payload['walk_ready_fsm_ids']}, expected the sorted "
        f"WALK_FSMS set {sorted(WALK_FSMS)}."
    )


def test_the_list_verb_names_the_transport_refusal_codes() -> None:
    """The list verb quotes both transport-level refusal codes.

    ``3102`` (send fail) and ``3104`` (timeout) are the codes a
    caller reading a refusal from a future ``_Call`` wrapper would
    see when the transport itself fails, independent of whether the
    api id is valid. The lookup names both so the caller knows what
    to expect on the two transport-failure paths.
    """
    payload = _call(g1_list_loco_rpc_apis)
    surfaced = {entry["code"] for entry in payload["transport_refusals"]}
    expected = set(_TRANSPORT_REFUSAL_CODES)
    assert surfaced == expected, (
        f"g1_list_loco_rpc_apis surfaced transport refusal codes {sorted(surfaced)}, expected {sorted(expected)}."
    )
    for entry in payload["transport_refusals"]:
        assert entry["text"] == ERR_CODES[entry["code"]], (
            f"transport refusal {entry} does not quote the shared ERR_CODES text; the two sides would drift."
        )


def test_the_admit_verb_admits_every_read_api() -> None:
    """Every id in :data:`_READ_APIS` reaches the admitted path.

    The five read api ids (``7001`` .. ``7005``) are the ones the
    neon bundle's ``read_fsm_id`` .. ``read_stand_height`` helpers
    pin; the admit verb returns the descriptor with
    ``kind="read"`` and ``touches_motion_gate=False``, matching the
    :func:`g1_list_loco_rpc_apis` shape.
    """
    for api_id in _READ_APIS:
        payload = _call(g1_loco_rpc_api_admits, api_id=api_id)
        assert payload["status"] == "success", f"admit verb refused api_id={api_id} which is in _READ_APIS: {payload}"
        assert payload["api"]["api_id"] == api_id
        assert payload["api"]["kind"] == "read"
        assert payload["api"]["touches_motion_gate"] is False
        assert payload["walk_ready_fsm_ids"] == sorted(WALK_FSMS)


def test_the_admit_verb_admits_every_write_api() -> None:
    """Every id in :data:`_WRITE_APIS` reaches the admitted path.

    Only ``7103`` today; the admit verb returns the descriptor with
    ``kind="write"`` and ``touches_motion_gate=True`` so a caller
    comparing against the walk-ready set sees the gate applies.
    """
    for api_id in _WRITE_APIS:
        payload = _call(g1_loco_rpc_api_admits, api_id=api_id)
        assert payload["status"] == "success", f"admit verb refused api_id={api_id} which is in _WRITE_APIS: {payload}"
        assert payload["api"]["api_id"] == api_id
        assert payload["api"]["kind"] == "write"
        assert payload["api"]["touches_motion_gate"] is True


def test_the_admit_verb_refuses_a_missing_argument() -> None:
    """A missing ``api_id`` is refused with the shared invalid code.

    The neon bundle's helpers never call ``_loco_call`` without an
    api id; a caller reaching the admit verb without one is asking
    an undecidable question. The verb refuses with the same
    ``7404`` code an unknown api id would trigger so the two miss
    paths share a code.
    """
    payload = _call(g1_loco_rpc_api_admits)
    assert payload["status"] == "error"
    assert payload["refusal_code"] == _INVALID_API_CODE
    assert payload["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
    assert "api_id not supplied" in payload["reason"]


def test_the_admit_verb_refuses_a_bool_argument() -> None:
    """A ``bool`` argument is refused before the ``int`` lookup runs.

    Python's ``bool`` is a subclass of ``int``: ``True`` compares
    equal to ``1``, ``False`` to ``0``. If the admit verb accepted
    ``bool`` silently, a caller passing ``True`` would look up ``1``
    (not in the admitted set) and receive a confusing refusal.
    Refusing at the type boundary makes the failure name the type
    mistake instead.
    """
    for arg in (True, False):
        payload = _call(g1_loco_rpc_api_admits, api_id=arg)
        assert payload["status"] == "error"
        assert payload["refusal_code"] == _INVALID_API_CODE
        assert "bool" in payload["reason"]


def test_the_admit_verb_refuses_a_non_integer_argument() -> None:
    """A non-integer non-bool argument is refused with a type reason.

    The neon bundle's ``_loco_call(api_id, payload)`` types the api
    id as ``int``; a caller passing a string, float or other type
    reaches the admit verb by mistake. The verb refuses with the
    same ``7404`` code so the caller sees the same refusal shape as
    on an unknown id, with a reason string that names the type
    mismatch.
    """
    for arg in ("7001", 7001.0, [7001], None):
        # ``None`` is the missing-argument path already covered above;
        # the other three are the type-refusal path.
        if arg is None:
            continue
        payload = _call(g1_loco_rpc_api_admits, api_id=arg)  # type: ignore[arg-type]
        assert payload["status"] == "error"
        assert payload["refusal_code"] == _INVALID_API_CODE
        assert "not an int" in payload["reason"], f"refusal for {arg!r} did not name the type mismatch: {payload}"


def test_the_admit_verb_refuses_an_unknown_integer() -> None:
    """An integer outside :data:`_LOCO_RPC_APIS` is refused.

    The neon bundle observed six api ids; every other integer is
    outside the admitted set and reaches the SDK's own handler
    boundary. The verb refuses at the type boundary with the same
    ``7404`` code the SDK's handler would surface so the two sides
    quote the same text.
    """
    # Pick an integer safely outside the admitted set: the maximum
    # admitted id plus one, which is never going to collide with a
    # future widen because the neon bundle groups these api ids in
    # the 7000s and any new id would land in the 7001-7999 range.
    unknown = max(_LOCO_RPC_APIS) + 10_000
    payload = _call(g1_loco_rpc_api_admits, api_id=unknown)
    assert payload["status"] == "error"
    assert payload["refusal_code"] == _INVALID_API_CODE
    assert payload["refusal_text"] == ERR_CODES[_INVALID_API_CODE]
    assert str(unknown) in payload["reason"]


def test_the_admit_verb_names_the_walk_ready_fsm_set_on_admit() -> None:
    """On admit the verb quotes :data:`WALK_FSMS` for the gate-set half.

    Mirrors :func:`test_the_list_verb_names_the_walk_ready_fsm_set`:
    a caller comparing an intended write-side api call against both
    the api-id set and the fsm set reads both from the same payload.
    """
    # Any admitted id works here; the walk-ready set is a top-level
    # field on the admit-path envelope, not conditional on kind.
    api_id = min(_LOCO_RPC_APIS)
    payload = _call(g1_loco_rpc_api_admits, api_id=api_id)
    assert payload["status"] == "success"
    assert payload["walk_ready_fsm_ids"] == sorted(WALK_FSMS)
