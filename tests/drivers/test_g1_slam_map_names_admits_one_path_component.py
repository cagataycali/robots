"""The SLAM map-name lookup admits one path component and refuses the bundle's escape.

:mod:`strands_robots.tools.g1.g1_slam_map_names` ports the containment
decision behind the neon bundle's ``g1_slam_save`` / ``g1_slam_load``
(``cagataycali/neon-the-g1/tools/g1_slam.py``, ``_SlamRunner._safe_map_path``)
as two agent-facing verbs. The port deliberately does *not* reproduce the
bundle's rule: the bundle joins the caller's name onto ``~/maps``, resolves
the result and tests containment by string prefix, which admits a sibling
directory whose name begins with the root's name. The module decides the
name before any join instead, so containment is structural.

The cells here grade three things:

1. **The snapshot.** The root spelling, the suffixes and the refused sets
   are read off the module's own constants rather than restated, so a widen
   or narrow surfaces here as a shape change instead of as a table this
   file would have to be remembered into.
2. **The correction.** :func:`test_the_bundle_prefix_rule_admits_an_escape_this_lookup_refuses`
   reimplements the bundle's check locally, against a temporary root, and
   pins both halves: the prefix rule admits ``../<root>-evil/pwn`` and the
   resulting path is outside the root, while the verb refuses that name.
   That cell is the reason the port exists; without it the corrected rule
   is indistinguishable from a transcription.
3. **The structural guarantee.** Every admitted name joins to a file
   directly inside an arbitrary root, checked by path algebra rather than
   by writing anything.

Two things these cells do not pin:

* The filesystem's answer. Nothing here creates ``~/maps``, writes a map
  or reads one: the verbs make no filesystem call, which is the property
  under test in :func:`test_the_decision_reads_no_filesystem_state`, and a
  test that touched the disk to check it would be asserting about the disk.
* Whether an admitted name is free, short enough for the destination's
  per-name byte limit, or distinct from an existing map on a
  case-insensitive volume. The module's docstring names all three as reads
  of the destination it does not make, and its returned payload claims none
  of them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.g1.g1_slam_map_names import (
    _LISTED_SUFFIXES,
    _MAP_ROOT_SPEC,
    _MAP_SUFFIX,
    _REFUSAL_TEXT,
    _REFUSED_CHARACTER_LABELS,
    _REFUSED_CHARACTERS,
    _REFUSED_COMPONENTS,
    _RULES,
    g1_list_slam_map_name_rules,
    g1_slam_map_name_admits,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function when
    called in-process, but a caller cannot rely on that: the wrapper's
    contract is that it returns the wrapped function's return value
    verbatim. This helper is where a shape drift would surface once,
    rather than at every call site.
    """
    return tool(**kwargs)


def _bundle_admits(root: Path, name: str) -> Path | None:
    """The neon bundle's ``_safe_map_path``, verbatim but parameterised on its root.

    ``cagataycali/neon-the-g1/tools/g1_slam.py`` reads its root off a
    module-level ``MAPS_DIR = Path.home() / "maps"``; the root is an
    argument here so the cell that grades the rule can point it at a
    ``tmp_path`` and assert about a real filesystem layout without
    depending on the runner's home directory. The body is otherwise
    unchanged - the join, the ``resolve``, and the ``str(...).startswith``
    containment test - because a paraphrase would grade a rule the bundle
    does not have.
    """
    try:
        path = (root / f"{name}{_MAP_SUFFIX}").resolve()
    except Exception:
        return None
    if not str(path).startswith(str(root.resolve())):
        return None
    return path


def test_the_import_pulls_no_sdk_or_slam_stack() -> None:
    """The tool module is loadable without the SDK and without the SLAM stack.

    Every file under :mod:`strands_robots.tools.g1` must import with
    ``unitree_sdk2py`` absent (refs strands-labs/robots#358). This module
    carries a second half: the bundle's ``g1_slam`` module imports
    ``numpy`` at module scope and ``open3d`` behind a ``try``, and drives
    ``kiss_icp`` in its worker, none of which a name decision needs. A
    lookup that pulled any of them would make the decision unavailable on
    exactly the hosts that have to author a name before the SLAM extra is
    installed.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_slam_map_names")
    after = set(sys.modules)
    leaked = {
        name
        for name in after - before
        if any(probe in name.lower() for probe in ("unitree", "open3d", "kiss_icp", "numpy"))
    }
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_slam_map_names imports pulled {leaked}. The rule for "
        "this package is that the SDK loads only inside function bodies, and this module's "
        "decision needs no array or registration library at all "
        "(refs strands-labs/robots#358)."
    )


def test_the_map_root_is_named_unexpanded() -> None:
    """The root is the ``~/maps`` spelling, not a resolved home directory.

    The module states the root without expanding it so the reported rule
    set does not vary with ``$HOME``: a headless CI runner and an
    operator's shell have to report the same field, and an expansion here
    would also reintroduce the resolved root the bundle's joined-then-
    tested check needed.
    """
    assert _MAP_ROOT_SPEC == "~/maps"
    assert _MAP_ROOT_SPEC.startswith("~"), "an expanded root would vary with the deciding host's home"
    assert not Path(_MAP_ROOT_SPEC).is_absolute()


def test_the_suffixes_match_the_bundle_writer_and_listing() -> None:
    """One suffix is written; two are listed, in the bundle's chaining order.

    ``save_map`` writes ``.npz`` only, while ``list_maps`` chains
    ``glob("*.npz")`` ahead of ``glob("*.npy")`` and dedupes on the stem -
    so the order is what decides which of two same-stem files the listing
    reports, and it is pinned rather than sorted.
    """
    assert _MAP_SUFFIX == ".npz"
    assert _LISTED_SUFFIXES == (".npz", ".npy")
    assert _LISTED_SUFFIXES[0] == _MAP_SUFFIX
    assert isinstance(_LISTED_SUFFIXES, tuple), "a mutable listing order could be reordered by a caller"


def test_the_refused_characters_cover_both_separators_and_nul() -> None:
    """Both path separators and NUL are refused on every platform.

    Keyed on literals rather than on ``os.sep`` / ``os.altsep`` because a
    map name is authored on one host and used on others: a rule that read
    the deciding host's separator would admit ``a\\b`` on Linux and then
    have it name a directory the first time the same string reached a
    Windows path.
    """
    assert _REFUSED_CHARACTERS == ("/", "\\", "\x00")
    assert len(_REFUSED_CHARACTER_LABELS) == len(_REFUSED_CHARACTERS)


def test_the_refused_character_labels_are_printable_ascii() -> None:
    """The labels a payload quotes carry no NUL and no non-ASCII.

    A returned payload is text an agent reads back. Quoting the refused
    characters themselves would put a NUL inside it, which truncates the
    rest of the string for any consumer that treats the payload as a C
    string - so the labels are what the verbs report, and they have to be
    safe to report.
    """
    for label in _REFUSED_CHARACTER_LABELS:
        assert label.isascii(), f"{label!r} is not ASCII"
        assert "\x00" not in label
        assert label.strip() == label


def test_the_refused_components_are_the_two_traversal_tokens() -> None:
    """Exactly ``.`` and ``..``, held immutably.

    These are the only two strings that name a directory rather than a
    file in every path context; the set is a ``frozenset`` so a caller
    holding the module cannot widen the admitted set by mutating it.
    """
    assert _REFUSED_COMPONENTS == frozenset({".", ".."})
    assert isinstance(_REFUSED_COMPONENTS, frozenset)


def test_the_rule_ids_are_unique_and_ordered_as_applied() -> None:
    """Each rule is named once, and the order is the application order.

    The order is load-bearing: the type rules precede the content rules
    because a non-string has no characters to scan, and ``name_is_not_a_bool``
    precedes ``name_is_a_string`` so a caller who passed a flag reads that
    word rather than a general type refusal.
    """
    ids = [rule["rule"] for rule in _RULES]
    assert len(ids) == len(set(ids)), f"a rule id is reported twice: {ids}"
    assert ids == [
        "name_is_required",
        "name_is_not_a_bool",
        "name_is_a_string",
        "name_is_not_empty",
        "name_is_one_path_component",
        "name_is_not_a_dot_component",
    ]


def test_every_rule_descriptor_carries_the_three_fields() -> None:
    """A rule names its id, what it requires, and the consequence of admitting a miss.

    ``why`` is the field an agent quotes when it has to explain a refusal
    to an operator, so an empty one is a rule that cannot be explained.
    """
    for rule in _RULES:
        assert set(rule) == {"rule", "requires", "why"}
        for field, value in rule.items():
            assert value.strip(), f"rule {rule['rule']!r} has an empty {field}"
            assert value.isascii(), f"rule {rule['rule']!r} has a non-ASCII {field}"


def test_the_listing_reports_the_snapshot() -> None:
    """The listing verb reports every constant, and the rule count matches the list."""
    payload = _call(g1_list_slam_map_name_rules)
    assert payload["status"] == "success"
    assert payload["map_root"] == _MAP_ROOT_SPEC
    assert payload["map_suffix"] == _MAP_SUFFIX
    assert payload["listed_suffixes"] == list(_LISTED_SUFFIXES)
    assert payload["refused_characters"] == list(_REFUSED_CHARACTER_LABELS)
    assert payload["refused_components"] == sorted(_REFUSED_COMPONENTS)
    assert payload["count"] == len(_RULES)
    assert [rule["rule"] for rule in payload["rules"]] == [rule["rule"] for rule in _RULES]
    assert payload["refusal_text"] == _REFUSAL_TEXT


def test_the_listing_reports_labels_rather_than_the_characters() -> None:
    """No raw separator-or-NUL character reaches the returned payload.

    The NUL half is the one that matters: a payload carrying it is
    truncated for any consumer that reads it as a C string, and the
    truncation is silent.
    """
    payload = _call(g1_list_slam_map_name_rules)
    for reported in payload["refused_characters"]:
        assert "\x00" not in reported
    assert "\x00" not in repr(payload["rules"])


def test_the_listing_hands_out_copies() -> None:
    """A caller mutating the returned rules does not widen the module's own set."""
    first = _call(g1_list_slam_map_name_rules)
    first["rules"][0]["rule"] = "mutated"
    first["listed_suffixes"].append(".bogus")
    second = _call(g1_list_slam_map_name_rules)
    assert second["rules"][0]["rule"] == "name_is_required"
    assert second["listed_suffixes"] == list(_LISTED_SUFFIXES)
    assert _RULES[0]["rule"] == "name_is_required"


@pytest.mark.parametrize(
    "name",
    [
        "office",
        "office-2",
        "office_2",
        "lab.floor2",
        "Map",
        "map",
        "a",
        "...",
        ".hidden",
        "office.npz",
        "office.npy",
        "  spaced  ",
        "-leading-dash",
    ],
)
def test_an_ordinary_stem_is_admitted(name: str) -> None:
    """A single component is admitted, and the payload names the file it becomes.

    ``.hidden`` is admitted deliberately: ``Path.glob`` matches a leading
    dot, so the bundle's listing reports it and the name round-trips.
    ``office.npz`` is admitted too - it becomes ``office.npz.npz``, whose
    stem is ``office.npz``, so that name round-trips as well.
    """
    payload = _call(g1_slam_map_name_admits, name=name)
    assert payload["status"] == "success", payload
    assert payload["name"] == name
    assert payload["filename"] == f"{name}{_MAP_SUFFIX}"
    assert payload["map_root"] == _MAP_ROOT_SPEC
    assert payload["listed_suffixes"] == list(_LISTED_SUFFIXES)


@pytest.mark.parametrize(
    "name",
    ["office", "lab.floor2", ".hidden", "office.npz", "..."],
)
def test_an_admitted_name_round_trips_through_the_listing(name: str) -> None:
    """The stem the listing would report for an admitted name is that name.

    Path algebra only - nothing is written. The bundle's listing reports
    ``p.stem`` for each globbed file and a caller loads what it read back,
    so a name whose file has a different stem is a map that is listed
    under a name it cannot be loaded by. That is exactly what the empty
    name does, which is why it is refused rather than admitted.
    """
    assert _call(g1_slam_map_name_admits, name=name)["status"] == "success"
    assert Path(f"{name}{_MAP_SUFFIX}").stem == name


@pytest.mark.parametrize(
    "name",
    ["office", "lab.floor2", ".hidden", "...", "office.npz"],
)
def test_an_admitted_name_joins_directly_inside_any_root(name: str, tmp_path: Path) -> None:
    """Containment is structural: the joined path's parent is the root itself.

    Checked against a ``tmp_path`` root rather than ``~/maps`` to make the
    point that the guarantee is a property of the name and not of one
    particular directory. ``is_relative_to`` alone would also accept a
    subdirectory, so the parent is compared as well.
    """
    root = tmp_path / "maps"
    payload = _call(g1_slam_map_name_admits, name=name)
    joined = root / payload["filename"]
    assert joined.parent == root
    assert joined.is_relative_to(root)
    assert len(joined.relative_to(root).parts) == 1


def test_a_missing_name_is_refused_decidably() -> None:
    """No default name; the refusal names the rule and the bundle's verdict."""
    payload = _call(g1_slam_map_name_admits)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_required"
    assert payload["refusal_text"] == _REFUSAL_TEXT
    assert "name is required" in payload["reason"]


@pytest.mark.parametrize("name", [True, False])
def test_a_bool_is_refused_ahead_of_the_general_type_rule(name: bool) -> None:
    """A flag is reported as a flag, because ``str(True)`` is an ordinary stem.

    A general type refusal would be true but unhelpful here: the caller's
    mistake is that the argument is a flag, and the consequence is a map
    file named ``True.npz``, so the refusal says so.
    """
    payload = _call(g1_slam_map_name_admits, name=name)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_not_a_bool"
    assert "bool" in payload["reason"]
    assert str(name) in payload["reason"]


@pytest.mark.parametrize("name", [0, 1, 2.5, ["office"], {"name": "office"}, Path("office")])
def test_a_non_string_is_refused(name: Any) -> None:
    """A path object or an index is reported rather than coerced.

    Coercing would pick the spelling the containment rule then refuses -
    ``str(Path("a/b"))`` is ``"a/b"`` - and report the wrong rule for the
    caller's actual mistake.
    """
    payload = _call(g1_slam_map_name_admits, name=name)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_a_string"


def test_the_empty_name_is_refused_and_the_refusal_names_the_round_trip() -> None:
    """The empty name is the one candidate the listing cannot round-trip.

    It makes the file ``.npz``, and ``Path(".npz").stem`` is ``".npz"``
    rather than ``""`` - so the listing reports the name ``.npz``, and
    loading that name looks for ``.npz.npz``. Both halves are asserted
    here: the path algebra that makes it a round-trip break, and the
    refusal that reports it.
    """
    assert Path(_MAP_SUFFIX).stem == _MAP_SUFFIX
    assert Path(f"{_MAP_SUFFIX}{_MAP_SUFFIX}").stem == _MAP_SUFFIX
    payload = _call(g1_slam_map_name_admits, name="")
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_not_empty"
    assert _MAP_SUFFIX in payload["reason"]


@pytest.mark.parametrize(
    "name",
    [
        "sub/dir",
        "/absolute",
        "../sibling",
        "../../etc/passwd",
        "windows\\path",
        "trailing/",
        "nul\x00byte",
    ],
)
def test_a_name_that_is_more_than_one_component_is_refused(name: str) -> None:
    """A separator or a NUL makes the string a path, and a path decides its own landing site."""
    payload = _call(g1_slam_map_name_admits, name=name)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_one_path_component"
    assert payload["refusal_text"] == _REFUSAL_TEXT


def test_a_refusal_reason_escapes_a_nul_rather_than_carrying_it() -> None:
    """The reason quotes the offending name through ``repr``, so a NUL is escaped.

    Interpolating the name raw would put the NUL in the payload the caller
    reads back, truncating everything the refusal says after it.
    """
    payload = _call(g1_slam_map_name_admits, name="nul\x00byte")
    assert "\x00" not in payload["reason"]
    assert "\\x00" in payload["reason"]


@pytest.mark.parametrize("name", [".", ".."])
def test_a_dot_component_is_refused(name: str) -> None:
    """Both traversal tokens are refused even though the flat writer makes them inert.

    ``.`` becomes the filename ``..npz`` under a writer that appends a
    suffix, so admitting it breaks nothing today. It is refused so that an
    admitted name is safe to interpolate as a component too - a writer
    that gave each map its own directory would escape on exactly these two
    strings.
    """
    payload = _call(g1_slam_map_name_admits, name=name)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_not_a_dot_component"


def test_every_rule_in_the_listing_is_reachable() -> None:
    """Each advertised rule is the one some candidate actually gets refused by.

    Derived by driving the verb, not by restating the table: a rule that
    no input can reach is documentation for a refusal that never happens,
    and a rule reachable only by an input the listing does not describe is
    the same drift in the other direction.
    """
    candidates: list[Any] = [None, True, 0, "", "sub/dir", "."]
    reached = set()
    for candidate in candidates:
        payload = (
            _call(g1_slam_map_name_admits) if candidate is None else _call(g1_slam_map_name_admits, name=candidate)
        )
        assert payload["status"] == "error", candidate
        reached.add(payload["rule"])
    assert reached == {rule["rule"] for rule in _RULES}


def test_every_refusal_carries_the_bundle_verdict_and_the_issue_reference() -> None:
    """One verdict string for one verdict, and every reason cites the porting issue."""
    for candidate in (None, True, 0, "", "sub/dir", "."):
        payload = (
            _call(g1_slam_map_name_admits) if candidate is None else _call(g1_slam_map_name_admits, name=candidate)
        )
        assert payload["refusal_text"] == _REFUSAL_TEXT
        assert "strands-labs/robots#358" in payload["reason"]
        assert payload["reason"].isascii()


def test_the_bundle_prefix_rule_admits_an_escape_this_lookup_refuses(tmp_path: Path) -> None:
    """The corrected rule, pinned against the rule it replaces.

    The bundle joins the name onto its root, resolves, and tests
    containment with ``str(path).startswith(str(root))``. A sibling
    directory whose name begins with the root's name satisfies that
    prefix, so the escape is admitted and the write lands outside the
    root. This lookup refuses the same name on the containment rule,
    before any join.
    """
    root = tmp_path / "maps"
    escape = f"../{root.name}-evil/pwn"

    landed = _bundle_admits(root, escape)
    assert landed is not None, "the bundle's prefix rule admits this name"
    assert not landed.is_relative_to(root.resolve()), "and the admitted name lands outside the root"
    assert landed == (tmp_path / f"{root.name}-evil" / f"pwn{_MAP_SUFFIX}")

    payload = _call(g1_slam_map_name_admits, name=escape)
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_one_path_component"


def test_the_bundle_prefix_rule_admits_a_subdirectory_the_listing_cannot_report(tmp_path: Path) -> None:
    """The second half of the same defect: contained, but not listable.

    ``sub/dir`` stays inside the root and is therefore admitted by the
    prefix test, but the bundle's listing globs the top level only
    (``MAPS_DIR.glob("*.npz")``), so the map is saved and then invisible
    to the caller who saved it. Asserted as path algebra - the joined
    path is two components deep - rather than by writing a file.
    """
    root = tmp_path / "maps"

    landed = _bundle_admits(root, "sub/dir")
    assert landed is not None
    assert landed.is_relative_to(root.resolve())
    assert len(landed.relative_to(root.resolve()).parts) == 2, "so a top-level glob does not report it"

    payload = _call(g1_slam_map_name_admits, name="sub/dir")
    assert payload["status"] == "error"
    assert payload["rule"] == "name_is_one_path_component"


def test_the_decision_reads_no_filesystem_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The verbs answer identically whatever ``$HOME`` is and whether ``~/maps`` exists.

    The bundle's root is ``Path.home() / "maps"``, so a decision that
    consulted it would move with the environment. Both verbs are driven
    under two different homes - one with a populated ``maps`` directory,
    one with no home directory at all - and required to return byte-equal
    payloads.
    """
    populated = tmp_path / "home-with-maps"
    (populated / "maps").mkdir(parents=True)
    (populated / "maps" / f"office{_MAP_SUFFIX}").write_bytes(b"")

    monkeypatch.setenv("HOME", str(populated))
    monkeypatch.setenv("USERPROFILE", str(populated))
    with_maps = (_call(g1_list_slam_map_name_rules), _call(g1_slam_map_name_admits, name="office"))

    monkeypatch.setenv("HOME", str(tmp_path / "home-that-is-absent"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home-that-is-absent"))
    without_maps = (_call(g1_list_slam_map_name_rules), _call(g1_slam_map_name_admits, name="office"))

    assert with_maps == without_maps
    assert with_maps[1]["status"] == "success"
