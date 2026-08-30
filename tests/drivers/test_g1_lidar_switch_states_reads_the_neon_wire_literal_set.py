"""The LiDAR switch state lookup names what the neon switch verb writes.

The Unitree G1 exposes a control channel for the head-mounted Livox
Mid-360 on the ``rt/utlidar/switch`` DDS topic: publishing an
``std_msgs::String_`` message whose ``data`` field is the literal
ASCII ``"ON"`` powers the LiDAR on, and the literal ``"OFF"``
powers it off. The neon bundle's ``g1_lidar_switch`` verb
(``cagataycali/neon-the-g1/tools/g1_lidar.py``) writes those two
strings verbatim from a Python ``bool`` argument, and no third
literal is honoured by the LiDAR firmware today. The
:mod:`strands_robots.tools.g1.g1_lidar_switch_states` module
snapshots that two-entry admission set as an agent-facing lookup and
exposes two verbs -
:func:`g1_list_lidar_switch_states` (name the whole set) and
:func:`g1_lidar_switch_state_admits` (decide one query) - so a
caller planning a future driver-side switch wrapper can name the
wire strings decidably before the write is attempted. The tests
here fix that contract without pulling the SDK or the DDS bus: the
module is loadable on a host without ``unitree_sdk2py`` installed,
so a headless CI runner and Thor before an office bring-up can read
the wire-literal set without triggering an import-time refusal.

Two things this file's cells deliberately do not pin:

* The runtime switch write. The neon bundle's ``g1_lidar_switch``
  publishes ``rt/utlidar/switch`` through a live DDS publisher; that
  write is a live bus reach that would compete with the driver's own
  subscribers for the same topic, and it is out of scope for this
  lookup. A caller comparing an intended write against both
  conditions (membership + bus reachable) reaches the driver after
  this verb admits the literal. This file does not exercise the
  publisher.
* The current LiDAR power state. The driver's ``_on_lidar_state``
  handler answers the "on or off right now" question through the
  sibling ``g1_lidar_state`` verb; the wire-literal admission set is
  a *write-side* answer naming what the topic accepts, not what the
  LiDAR is doing right now. This file does not read the state cache.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_lidar_switch_states import (
    _INVALID_SWITCH_STATE_CODE,
    _LIDAR_SWITCH_ROLES,
    _LIDAR_SWITCH_STATES,
    g1_lidar_switch_state_admits,
    g1_list_lidar_switch_states,
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
    runner and Thor before an office bring-up. The switch-state
    snapshot is a string tuple; no SDK submodule should load on the
    import path.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_switch_states")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_lidar_switch_states imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads on driver ``connect_eagerly``, not on tool import."
    )


def test_the_import_pulls_no_cyclonedds_module() -> None:
    """The tool module is loadable without CycloneDDS bindings.

    The neon bundle's ``g1_lidar_switch`` reaches CycloneDDS through
    the shared publisher path; a lookup that only names the wire
    literals must not pull the DDS binding on import. A machine
    without ``cyclonedds`` installed must be able to read the
    admitted set without triggering an ImportError at tool discovery.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_lidar_switch_states")
    after = set(sys.modules)
    dds_prefixes = ("cyclonedds", "cyclonedds_py")
    leaked = {
        name
        for name in after - before
        if any(name == prefix or name.startswith(prefix + ".") for prefix in dds_prefixes)
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_lidar_switch_states imports pulled "
        f"CycloneDDS submodules: {leaked}. The switch-state snapshot is a "
        "string tuple; the DDS binding loads on the write path, not on "
        "tool import."
    )


def test_the_snapshot_is_an_immutable_tuple() -> None:
    """The snapshot is a ``tuple`` so a caller cannot mutate it in place.

    The neon source writes the two wire strings as bare string
    literals inside the switch branch (``"ON"``/``"OFF"``); a list
    on the snapshot side would let a caller reading
    :data:`_LIDAR_SWITCH_STATES` mutate the module state and drift
    the lookup out of sync with the neon source. The tuple type is
    a defensive contract the tests fix.
    """
    assert isinstance(_LIDAR_SWITCH_STATES, tuple), (
        f"_LIDAR_SWITCH_STATES is not a tuple: {type(_LIDAR_SWITCH_STATES).__name__}. "
        "A mutable snapshot would let a caller drift the lookup out of sync with the neon source."
    )


def test_the_snapshot_names_the_two_wire_literals_in_neon_truth_table_order() -> None:
    """The snapshot writes ``"ON"`` first and ``"OFF"`` second.

    The neon source's ``g1_lidar_switch`` verb defaults ``on=True``,
    which selects ``"ON"``; the ``False`` branch selects ``"OFF"``.
    A caller reading the payload without arguments lands on the
    default branch, and the ordering pins which literal that
    branch reaches. A silent reorder would flip which literal the
    argumentless default selects, so pinning the exact tuple
    surfaces that as a shape change rather than as a quiet
    semantic swap.
    """
    assert _LIDAR_SWITCH_STATES == ("ON", "OFF"), (
        f"lidar switch state order drifted from the neon source: {_LIDAR_SWITCH_STATES}. "
        "The neon ``g1_lidar_switch`` verb defaults ``on=True`` which selects the first "
        "literal; a reorder is a semantic change to the argumentless default."
    )


def test_the_snapshot_covers_two_states() -> None:
    """The snapshot names exactly two wire literals.

    The LiDAR firmware admits ``"ON"`` and ``"OFF"`` today; a widen
    or narrow to the admission set is a firmware-side change that
    must be reflected here. Pinning the count surfaces a silent
    drift as a shape change rather than as a quiet addition.
    """
    assert len(_LIDAR_SWITCH_STATES) == 2, (
        f"expected 2 admitted switch states, got {len(_LIDAR_SWITCH_STATES)}: {_LIDAR_SWITCH_STATES}"
    )


def test_every_state_is_ascii_uppercase() -> None:
    """Every wire literal is ASCII uppercase.

    The LiDAR firmware admits the ASCII uppercase form only; a
    lowercase or mixed-case literal reaching the wire is silently
    dropped by the firmware today. A snapshot entry that lost its
    uppercase-ness would silently start refusing writes at the
    firmware boundary that this lookup was meant to admit at the
    caller boundary; pinning the case surfaces such a drift.
    """
    for state in _LIDAR_SWITCH_STATES:
        assert state.isascii(), f"switch state {state!r} contains non-ASCII characters"
        assert state.isupper(), (
            f"switch state {state!r} is not ASCII uppercase; the LiDAR firmware admits "
            "the uppercase form only, and a case-variant literal is silently dropped at the wire"
        )


def test_every_state_carries_a_role() -> None:
    """Every admitted state names a non-empty role label.

    The role is what the caller reads to classify the state
    (``power_on`` vs ``power_off``); an empty role would leave the
    caller reading a bare literal without context about what the
    write selects.
    """
    for state in _LIDAR_SWITCH_STATES:
        assert state in _LIDAR_SWITCH_ROLES, (
            f"state {state!r} has no role entry in _LIDAR_SWITCH_ROLES; every "
            "admitted literal must name what power selection it targets"
        )
        entry = _LIDAR_SWITCH_ROLES[state]
        assert "role" in entry, f"state {state!r} has no role field; every admitted state must name what it selects"
        assert isinstance(entry["role"], str) and entry["role"], f"state {state!r} has an empty role: {entry['role']!r}"


def test_the_two_roles_are_distinct() -> None:
    """Each admitted state plays a distinct role in the switch path.

    ``power_on`` and ``power_off`` are opposite ends of the same
    two-state toggle, and neither role is a synonym of the other.
    A snapshot that folded them into a single role would silently
    lose the on/off distinction the LiDAR firmware honours; pinning
    role uniqueness surfaces that as a shape change.
    """
    roles = [_LIDAR_SWITCH_ROLES[state]["role"] for state in _LIDAR_SWITCH_STATES]
    assert len(set(roles)) == len(roles), (
        f"switch state roles collided: {roles}. Each admitted literal must play a distinct role, "
        "so a caller reading the payload can tell which literal selects which power state."
    )


def test_every_state_carries_a_non_empty_description() -> None:
    """Every state's description text is non-empty prose.

    The description is what a caller reads when the ``role`` label
    is not enough context - what observable effect the write has,
    what the LiDAR does afterwards, whether the write is idempotent.
    An empty description would leave a caller with no runbook after
    the write.
    """
    for state in _LIDAR_SWITCH_STATES:
        description = _LIDAR_SWITCH_ROLES[state]["description"]
        assert isinstance(description, str) and description.strip(), (
            f"state {state!r} has an empty or whitespace-only description; every "
            "admitted literal must name what the write selects and its observable effect"
        )


def test_the_invalid_switch_state_code_is_in_the_shared_err_table() -> None:
    """The refusal code the verbs quote is in :data:`ERR_CODES`.

    A code the lookup quoted but the shared error table did not name
    would leave a caller reading the refusal without a decoded text;
    pinning membership against the shared table catches a lookup
    that invented its own code the driver's refusal would not name
    back.
    """
    assert _INVALID_SWITCH_STATE_CODE in ERR_CODES, (
        f"_INVALID_SWITCH_STATE_CODE={_INVALID_SWITCH_STATE_CODE} is not in ERR_CODES; the "
        "verbs quote both the code and the shared table's text, so a code outside the table "
        "would leave the refusal without a decoded name"
    )


def test_list_returns_success_status() -> None:
    """The list verb reports ``status=success`` on the read path.

    The verb reads only module-level constants; the ``status`` field
    is ``success`` unconditionally because no refusal path exists
    on the read side.
    """
    payload = _call(g1_list_lidar_switch_states)
    assert payload["status"] == "success", f"list payload status is not success: {payload}"


def test_list_reports_the_two_states_in_neon_order() -> None:
    """The list verb's ``ordered_states`` matches the snapshot.

    A drift between the list payload and the snapshot would let the
    admits verb accept a literal the list did not name (or refuse
    a literal the list did name); the test pins that the two
    surfaces agree on the same tuple.
    """
    payload = _call(g1_list_lidar_switch_states)
    assert payload["ordered_states"] == list(_LIDAR_SWITCH_STATES), (
        f"list payload ordered_states {payload['ordered_states']} drifted from snapshot {list(_LIDAR_SWITCH_STATES)}"
    )


def test_list_carries_every_snapshot_descriptor() -> None:
    """The list verb surfaces one descriptor per snapshot entry.

    The descriptor shape (``wire_literal``/``role``/``description``/
    ``is_default``) is the same shape the admits verb returns on
    admit; a shape drift would let one verb name a field the other
    did not, and a caller migrating from list to admits would find
    a missing key at runtime. Pinning that every list descriptor
    carries every field surfaces the drift here rather than at the
    caller.
    """
    payload = _call(g1_list_lidar_switch_states)
    assert len(payload["states"]) == len(_LIDAR_SWITCH_STATES), (
        f"list payload states count {len(payload['states'])} does not match snapshot count {len(_LIDAR_SWITCH_STATES)}"
    )
    for descriptor in payload["states"]:
        assert "wire_literal" in descriptor, f"list descriptor missing wire_literal: {descriptor}"
        assert "role" in descriptor, f"list descriptor missing role: {descriptor}"
        assert "description" in descriptor, f"list descriptor missing description: {descriptor}"
        assert "is_default" in descriptor, f"list descriptor missing is_default: {descriptor}"


def test_list_marks_on_as_the_default_and_off_as_non_default() -> None:
    """The ``is_default`` flag names the argumentless-default branch.

    The neon ``g1_lidar_switch`` verb defaults ``on=True``, which
    selects ``"ON"``; the ``is_default`` flag on that descriptor is
    ``True`` and every other descriptor's flag is ``False``. A
    caller reading the flag knows which literal a future
    driver-side wrapper's argumentless call would select without
    having to read the wrapper's source.
    """
    payload = _call(g1_list_lidar_switch_states)
    defaults = {d["wire_literal"]: d["is_default"] for d in payload["states"]}
    assert defaults == {"ON": True, "OFF": False}, (
        f"is_default flags drifted from the neon argumentless-default branch: {defaults}"
    )


def test_list_names_the_shared_refusal_code() -> None:
    """The list payload names the ``7404`` refusal a caller would face on a miss.

    Surfacing the refusal code on the read side lets a caller wire
    a decoded refusal message without a second call to the admits
    verb (which is the code path a caller reaches on a miss, not on
    a planning read).
    """
    payload = _call(g1_list_lidar_switch_states)
    refusal_codes = {entry["code"] for entry in payload["refusals"]}
    assert _INVALID_SWITCH_STATE_CODE in refusal_codes, (
        f"list payload refusals {payload['refusals']} does not name the shared "
        f"refusal code {_INVALID_SWITCH_STATE_CODE}"
    )


def test_admits_returns_success_for_every_admitted_literal() -> None:
    """The admits verb accepts every literal the list names.

    Round-tripping every snapshot entry through the admits verb
    surfaces a drift between the list and the admits sides: a
    literal named on the list but refused by admits would let a
    caller plan a write that the same module's admits check would
    reject, an inconsistency the tests catch here.
    """
    for state in _LIDAR_SWITCH_STATES:
        payload = _call(g1_lidar_switch_state_admits, wire_literal=state)
        assert payload["status"] == "success", f"admits refused {state!r} which the list names as admitted: {payload}"
        assert payload["state"]["wire_literal"] == state, (
            f"admits descriptor wire_literal drifted from argument {state!r}: {payload}"
        )


def test_admits_refuses_none_with_a_missing_argument_reason() -> None:
    """A ``None`` argument surfaces the "required" refusal.

    The default value on the admits verb is ``None`` (the tool
    schema does not know a default and cannot inject one), so a
    caller reaching the verb without arguments lands on the
    missing-argument branch. The reason string names the required
    set so the caller knows what to pass on the retry.
    """
    payload = _call(g1_lidar_switch_state_admits)
    assert payload["status"] == "error"
    assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE
    assert "required" in payload["reason"], (
        f"refusal reason does not name the missing-argument shape: {payload['reason']!r}"
    )


def test_admits_refuses_a_bool_argument_with_a_shape_reason() -> None:
    """A ``bool`` argument surfaces the "not a str" shape refusal.

    Python's ``bool`` is a subclass of ``int``, not of ``str``, so
    ``isinstance(True, str)`` is ``False`` and the non-string
    branch fires. The explicit bool refusal ahead of the general
    non-string branch keeps the reason string naming the bool
    subclass surprise a caller mistyping ``True`` as a switch
    argument would hit.
    """
    for arg in (True, False):
        payload = _call(g1_lidar_switch_state_admits, wire_literal=arg)
        assert payload["status"] == "error", f"bool {arg!r} was not refused: {payload}"
        assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE
        assert "bool" in payload["reason"].lower(), (
            f"refusal reason does not name the bool shape: {payload['reason']!r}"
        )


def test_admits_refuses_a_non_string_argument_with_a_shape_reason() -> None:
    """Non-string arguments (int, float, list, tuple) refuse decidably.

    The wire message's ``data`` field is a UTF-8 string; a
    non-string argument would not be identity-equal to any entry.
    Refusing the shape at the boundary keeps the reason string
    naming the type mistake a caller would otherwise see as an
    "unknown literal" refusal that hid the type error.
    """
    for arg in (1, 0, 3.14, [], (), object()):
        payload = _call(g1_lidar_switch_state_admits, wire_literal=arg)
        assert payload["status"] == "error", f"non-str {arg!r} was not refused: {payload}"
        assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE
        assert "str" in payload["reason"], (
            f"refusal reason for {arg!r} does not name the not-a-str shape: {payload['reason']!r}"
        )


def test_admits_refuses_an_empty_string_with_a_shape_reason() -> None:
    """The empty string surfaces the "empty" shape refusal.

    An empty string is a valid ``str`` shape but is never in the
    admitted set; catching it as a shape error rather than as an
    "unknown literal" keeps the reason string naming the empty
    input a caller passing a missing config value would hit.
    """
    payload = _call(g1_lidar_switch_state_admits, wire_literal="")
    assert payload["status"] == "error"
    assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE
    assert "empty" in payload["reason"].lower(), (
        f"refusal reason does not name the empty-string shape: {payload['reason']!r}"
    )


def test_admits_refuses_case_variant_literals_decidably() -> None:
    """Lowercase and mixed-case literals refuse rather than coerce.

    The LiDAR firmware admits the ASCII uppercase form only; a
    lowercase ``"on"`` or a mixed-case ``"On"`` reaching the wire
    is silently dropped by the firmware today. Coercing the case
    inside the admits verb would hide the wire drop as a
    caller-side success; refusing decidably surfaces the case
    mismatch to the caller before the write reaches the wire.
    """
    for arg in ("on", "off", "On", "Off", "oN", "oFF", "on ", " ON"):
        payload = _call(g1_lidar_switch_state_admits, wire_literal=arg)
        assert payload["status"] == "error", (
            f"case-variant {arg!r} was not refused; the firmware admits ASCII uppercase only: {payload}"
        )
        assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE


def test_admits_refuses_unknown_literals_with_the_admitted_set() -> None:
    """An unknown literal surfaces the admitted set in the refusal reason.

    A caller reading the refusal without also knowing the admitted
    set has to make a second call to :func:`g1_list_lidar_switch_states`
    to recover; naming the set in the reason string lets the
    caller retry with the right literal without the extra call.
    """
    payload = _call(g1_lidar_switch_state_admits, wire_literal="STANDBY")
    assert payload["status"] == "error"
    assert payload["refusal_code"] == _INVALID_SWITCH_STATE_CODE
    for state in _LIDAR_SWITCH_STATES:
        assert state in payload["reason"], (
            f"refusal reason does not name admitted literal {state!r}: {payload['reason']!r}"
        )


def test_admits_returns_the_same_descriptor_shape_as_list() -> None:
    """The admits verb's admit descriptor matches the list verb's.

    A shape drift between the two verbs would let a caller reading
    the admits descriptor find a field the list descriptor did not
    carry, or vice versa. Pinning both descriptors to the same
    keys surfaces the drift here.
    """
    list_payload = _call(g1_list_lidar_switch_states)
    for state in _LIDAR_SWITCH_STATES:
        admit_payload = _call(g1_lidar_switch_state_admits, wire_literal=state)
        list_descriptor = next(d for d in list_payload["states"] if d["wire_literal"] == state)
        assert admit_payload["state"] == list_descriptor, (
            f"admits descriptor for {state!r} drifted from list descriptor: "
            f"{admit_payload['state']} != {list_descriptor}"
        )
