"""Agent-facing lookup for the SLAM map names a save/load path can admit.

The neon bundle's ``cagataycali/neon-the-g1/tools/g1_slam.py`` keeps its
kiss-icp maps as one flat directory of ``.npz`` archives under
``~/maps``, and every one of its four name-taking verbs (``g1_slam_save``,
``g1_slam_load``, and the ``g1_slam_list_maps`` listing that reports the
names those two exchange) routes the caller's string through one private
containment check, ``_SlamRunner._safe_map_path``. That check is the whole
of the decision: an admitted name becomes a filesystem path a moment
later, and the name arrives from an agent (``g1_slam_save(name)`` is a
``@tool`` whose ``name`` argument is model-authored text). This module
ports that decision - and only the decision - as two agent-facing verbs:
:func:`g1_list_slam_map_name_rules` (name the whole rule set) and
:func:`g1_slam_map_name_admits` (decide one candidate). Refs
strands-labs/robots#358.

The port does not reproduce the bundle's rule
=============================================

The bundle joins first and tests the joined string afterwards::

    path = (MAPS_DIR / f"{name}.npz").resolve()
    if not str(path).startswith(str(MAPS_DIR.resolve())):
        return None

A resolved path is compared to the root by string prefix, and a *sibling
directory whose name merely begins with the root's name* satisfies that
prefix. Measured against ``MAPS_DIR = ~/maps`` on a host whose home
directory expands to ``$HOME``:

===========================  =============================  =================
``name``                     joined and resolved            bundle's verdict
===========================  =============================  =================
``ok``                       ``$HOME/maps/ok.npz``          admitted
``../maps-evil/pwn``         ``$HOME/maps-evil/pwn.npz``    admitted
``../maps.ssh/x``            ``$HOME/maps.ssh/x.npz``       admitted
``sub/dir``                  ``$HOME/maps/sub/dir.npz``     admitted
``../../etc/passwd``         (outside, no shared prefix)    refused
``/etc/passwd``              (outside, no shared prefix)    refused
===========================  =============================  =================

So the rule refuses the traversal that leaves the *parent* and admits the
one that lands beside the root: two of those admitted names write outside
``~/maps`` entirely, and the third writes into a subdirectory the
top-level ``MAPS_DIR.glob("*.npz")`` listing cannot report, so the map is
saved and then invisible to the caller who saved it.

This module decides the name *before* any join, on the name alone: a
candidate is admitted only when it is a single path component that is not
a traversal token. Containment is then structural rather than tested -
``root / f"{name}.npz"`` for an admitted ``name`` cannot be anywhere but
directly inside ``root``, whatever the root is and whatever the caller
resolves afterwards - which is why the port is not a transcription of the
check it replaces. :data:`_REFUSED_CHARACTERS` and
:data:`_REFUSED_COMPONENTS` are the two halves of that rule.

What this module is deliberately *not*
======================================

* A filesystem read. Nothing here stats, globs, opens or resolves
  anything: the verbs answer from module-level constants and the argument.
  So the answer does not depend on ``$HOME``, on whether ``~/maps``
  exists, on what it already holds, or on the platform - which is what
  lets a headless CI runner and an operator laptop decide a name
  identically, and what keeps the decision available before the directory
  is created. Whether an admitted name is *free* (no map of that name yet)
  is a directory read this module cannot make and does not claim.
* A save or load path. There is no map I/O in this package yet; when a
  driver-side SLAM verb lands it should reach this decision rather than
  re-derive one, which is the reason to land the rule ahead of the writer
  rather than inside it.
* A byte-length or encoding bound. A name of 300 ASCII characters is
  admitted here and refused by the kernel at ``open`` time with
  ``ENAMETOOLONG``, because the limit is per-filesystem (255 bytes on
  ext4, and it counts *encoded* bytes, so the admitted length in
  characters varies with the name's own script). A bound stated here
  would be a number this module cannot know, and stating it per-filesystem
  needs the filesystem, which is the read above.
* A case-collision answer. ``Map`` and ``map`` are two names here and one
  file on a case-insensitive volume, so a name admitted on both hosts can
  still collide on one of them. That is a property of the destination
  volume, not of the name.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The map directory the neon bundle keeps its archives in, stated as the
#: unexpanded spelling (``~/maps``) rather than as a resolved
#: :class:`~pathlib.Path`. Two reasons, and both are about the answer being
#: the same everywhere: expanding it would read ``$HOME`` at import, so the
#: rule set a headless runner reported and the one an operator's shell
#: reported would differ on a field that has nothing to do with the rules;
#: and a resolved root invites the joined-then-tested shape this module
#: exists to replace. The verbs never join it - it is surfaced so a caller
#: knows *which* directory the admitted name is a name in.
_MAP_ROOT_SPEC: str = "~/maps"

#: The suffix the bundle's ``save_map`` appends to an admitted name. A
#: caller passes the stem (``office``) and the file is ``office.npz``.
_MAP_SUFFIX: str = ".npz"

#: The suffixes the bundle's ``list_maps`` enumerates, in the order it
#: chains them. ``.npy`` is a legacy single-array map that the listing
#: still reports and ``save_map`` no longer writes; a stem carried by both
#: suffixes is reported once, for the ``.npz``, because the listing dedupes
#: on the stem and reaches ``.npz`` first. Surfaced so a caller reading a
#: name back off the listing knows the two spellings collapse.
_LISTED_SUFFIXES: tuple[str, ...] = (".npz", ".npy")

#: The characters that make a candidate more than one path component. Both
#: separators are refused on every platform, and stated as literals rather
#: than read off ``os.sep`` / ``os.altsep``, because a map name travels:
#: it is authored by an agent, written on the robot, listed on an
#: operator's laptop and asserted in CI, and a rule keyed on the deciding
#: host's own separator would admit ``a\b`` on Linux as an ordinary
#: filename and then name a directory the moment the same string reached a
#: Windows path. NUL is refused for the same reason it is not a filename
#: character anywhere: it terminates the path the C library is handed, so
#: the bytes after it are silently not part of the name.
_REFUSED_CHARACTERS: tuple[str, ...] = ("/", "\\", "\x00")

#: Printable labels for :data:`_REFUSED_CHARACTERS`, one per entry and in
#: the same order. The refusal strings and the rule listing quote these
#: rather than the characters themselves: a returned payload is text an
#: agent reads back, and putting a raw NUL in it makes the rest of the
#: string invisible to anything that treats the payload as a C string.
_REFUSED_CHARACTER_LABELS: tuple[str, ...] = (
    "'/' (forward slash)",
    "'\\' (backslash)",
    "NUL (0x00)",
)

#: The two strings that name a directory rather than a file in every path
#: context. The bundle's join makes them inert by accident - ``.`` becomes
#: the filename ``..npz`` - so refusing them changes no working call, and
#: it is what makes the admitted answer independent of *where* a caller
#: interpolates the name. A future writer that joins the name as its own
#: component (``root / name / "map.npz"``, one directory per map) would
#: otherwise escape on the same two strings the flat writer tolerated.
_REFUSED_COMPONENTS: frozenset[str] = frozenset({".", ".."})

#: The verdict string the bundle's ``save_map`` / ``load_map`` return to
#: the caller when ``_safe_map_path`` refuses (``{"ok": False, "error":
#: "invalid map name"}``). Quoted verbatim on every refusal here so a
#: caller that plans a call with this lookup and then makes it reads one
#: string for one verdict, rather than two wordings for the same refusal.
_REFUSAL_TEXT: str = "invalid map name"

#: The rule set, in the order :func:`g1_slam_map_name_admits` applies it.
#: Each entry names the ``rule`` id the refusal reports, what the rule
#: ``requires``, and ``why`` - the consequence of admitting what it
#: refuses, stated as the outcome a caller would get rather than as the
#: rule restated. The order is load-bearing and pinned: the type rules
#: precede the content rules because a non-string has no characters to
#: scan, and ``bool`` precedes the general type rule because
#: ``isinstance(True, int)`` is not the confusion being reported - a
#: caller who passed a flag wants to read that word in the refusal.
_RULES: tuple[dict[str, str], ...] = (
    {
        "rule": "name_is_required",
        "requires": "A name argument is passed.",
        "why": (
            "A missing name is refused decidably rather than defaulted. There is no map "
            "this lookup could name on the caller's behalf, and a default would be a name "
            "the caller did not choose that a later save would overwrite."
        ),
    },
    {
        "rule": "name_is_not_a_bool",
        "requires": "The name is not a bool.",
        "why": (
            "str(True) is 'True', a perfectly ordinary stem, so a bool that reached a "
            "save path would create a map named for a flag. The refusal names the type "
            "so the caller sees the argument they passed, not the file it would make."
        ),
    },
    {
        "rule": "name_is_a_string",
        "requires": "The name is a str.",
        "why": (
            "A non-string has no path-component answer. An int map index or a Path is a "
            "caller mistake this lookup reports rather than coerces, because coercing "
            "picks the spelling (str(Path('a/b')) is 'a/b') the next rule would refuse."
        ),
    },
    {
        "rule": "name_is_not_empty",
        "requires": "The name is not the empty string.",
        "why": (
            "An empty name is the one candidate that breaks the listing round trip. It "
            "makes the file '.npz', whose stem is '.npz' rather than '', so the listing "
            "reports the name '.npz' and loading that name looks for '.npz.npz' - a map "
            "saved, listed under a name, and not loadable by it."
        ),
    },
    {
        "rule": "name_is_one_path_component",
        "requires": "The name carries no path separator and no NUL.",
        "why": (
            "This is the containment rule. A name with a separator is a path, and a path "
            "decides for itself where it lands: '../maps-evil/pwn' escapes a '~/maps' "
            "root that a string-prefix containment test still calls contained, and "
            "'sub/dir' stays inside but under a subdirectory the flat top-level listing "
            "cannot report. One component cannot do either."
        ),
    },
    {
        "rule": "name_is_not_a_dot_component",
        "requires": "The name is not '.' or '..'.",
        "why": (
            "Both name a directory rather than a file. A writer that appends a suffix "
            "makes them inert filenames ('..npz'), and one that joins the name as its "
            "own component does not, so refusing them keeps an admitted name safe to "
            "interpolate at either position."
        ),
    },
)


def _refuse(rule: str, reason: str) -> dict[str, Any]:
    """Build the refusal envelope every rule returns.

    One helper so the ``status`` / ``refusal_text`` / ``rule`` / ``reason``
    shape is identical across the six rules, and so a caller can key on
    ``rule`` without also parsing ``reason``: the id is the stable half and
    the prose is the readable one.

    Args:
        rule: The id of the refusing rule, one of the ``rule`` values
            :func:`g1_list_slam_map_name_rules` reports.
        reason: Why this argument was refused, naming the argument.

    Returns:
        The refusal dict, carrying the bundle's own verdict string in
        ``refusal_text``.
    """
    return {
        "status": "error",
        "refusal_text": _REFUSAL_TEXT,
        "rule": rule,
        "reason": reason,
    }


@tool
def g1_list_slam_map_name_rules() -> dict[str, Any]:
    """Return the rules a SLAM map name has to satisfy, and where such a name is a name.

    Read-only. No driver instance, no DDS, no SDK, no filesystem: every
    field is a module-level constant. Useful before ``g1_slam_save`` /
    ``g1_slam_load`` on the neon side, or before a future driver-side map
    verb here, so a caller can author a name that will be admitted instead
    of discovering the refusal at save time - and so an agent that has to
    *explain* a refusal reads the same ``why`` text this lookup refused on.

    Returns:
        A dict with ``status``; ``map_root`` naming the directory an
        admitted name is a name in (``~/maps``, unexpanded - see
        :data:`_MAP_ROOT_SPEC`); ``map_suffix`` naming the suffix a writer
        appends (``.npz``); ``listed_suffixes`` naming every suffix the
        bundle's listing enumerates, in its own chaining order;
        ``refused_characters`` naming the characters that make a candidate
        more than one component, as printable labels rather than as the
        characters themselves; ``refused_components`` naming the two
        traversal tokens, sorted; ``count`` naming the number of rules; a
        ``rules`` list of descriptors in application order, each carrying
        ``rule``, ``requires`` and ``why``; and ``refusal_text`` naming the
        single verdict string a refusal reports.
    """
    return {
        "status": "success",
        "map_root": _MAP_ROOT_SPEC,
        "map_suffix": _MAP_SUFFIX,
        "listed_suffixes": list(_LISTED_SUFFIXES),
        "refused_characters": list(_REFUSED_CHARACTER_LABELS),
        "refused_components": sorted(_REFUSED_COMPONENTS),
        "count": len(_RULES),
        "rules": [dict(rule) for rule in _RULES],
        "refusal_text": _REFUSAL_TEXT,
    }


@tool
def g1_slam_map_name_admits(name: str | None = None) -> dict[str, Any]:
    """Decide whether ``name`` is a SLAM map name a save or load path can take.

    Read-only. Compares one argument against the rule set
    :func:`g1_list_slam_map_name_rules` reports and returns the filename an
    admitted name becomes, or the refusing rule and the bundle's own
    ``invalid map name`` verdict on a miss. No driver instance, no DDS, no
    SDK, and no filesystem access: the decision reads module-level
    constants and the argument, which is what makes it answerable before
    ``~/maps`` exists and identical on every host.

    An admitted name is not the same as an admitted save. The name is
    guaranteed to land directly inside the map root and nowhere else, and
    to round-trip through the listing; it is *not* guaranteed to be unused
    (a save overwrites a map of the same name), to be short enough for the
    destination filesystem's per-name byte limit, or to be distinct from an
    existing map under a case-insensitive volume. Those three are reads of
    the destination, and this verb refuses to guess at them rather than
    report an admission it cannot support.

    Args:
        name: The map name to check - the stem a writer appends
            ``.npz`` to, not a path and not a filename. Must be a
            non-empty ``str`` that is one path component (no ``/``, no
            ``\\``, no NUL) and is not ``.`` or ``..``. ``bool`` is
            refused ahead of the general type rule because ``str(True)``
            is an ordinary stem and a flag that reached a save path would
            create a map named for it. A missing argument (``None``) is
            refused decidably rather than treated as a default.

    Returns:
        A dict with ``status``. On admit: ``name`` echoing the argument,
        ``filename`` naming what a writer creates (``<name>.npz``),
        ``map_root`` naming the directory it is created in, and
        ``listed_suffixes`` naming the suffixes the listing reports it
        under. On refuse: ``refusal_text`` carrying the bundle's verdict
        string, ``rule`` naming the refusing rule id (one of the
        ``rule`` values the listing verb reports), and ``reason``
        naming the argument and why it was refused.
    """
    if name is None:
        return _refuse(
            "name_is_required",
            "name is required; pass the stem of the map file (for example 'office') so "
            "the lookup is decidable. Refs strands-labs/robots#358.",
        )
    if isinstance(name, bool):
        return _refuse(
            "name_is_not_a_bool",
            f"name={name!r} is a bool; pass the map's stem as a str, because str({name!r}) "
            f"is {str(name)!r} and would name a map after a flag. "
            "Refs strands-labs/robots#358.",
        )
    if not isinstance(name, str):
        return _refuse(
            "name_is_a_string",
            f"name={name!r} is not a str; pass the map's stem as a str rather than a path "
            "object or an index. Refs strands-labs/robots#358.",
        )
    if not name:
        return _refuse(
            "name_is_not_empty",
            f"name is the empty string; it would create the file {_MAP_SUFFIX!r}, which the "
            f"listing reports under the name {_MAP_SUFFIX!r} and which loading that name "
            "cannot find. Pass a non-empty stem. Refs strands-labs/robots#358.",
        )
    for character, label in zip(_REFUSED_CHARACTERS, _REFUSED_CHARACTER_LABELS, strict=True):
        if character in name:
            return _refuse(
                "name_is_one_path_component",
                f"name={name!r} contains {label}; a map name is one path component, so it "
                f"names a file directly inside {_MAP_ROOT_SPEC} and cannot select another "
                "directory. Refs strands-labs/robots#358.",
            )
    if name in _REFUSED_COMPONENTS:
        return _refuse(
            "name_is_not_a_dot_component",
            f"name={name!r} names a directory rather than a file; pass the stem of a map "
            f"file instead of one of {sorted(_REFUSED_COMPONENTS)}. "
            "Refs strands-labs/robots#358.",
        )
    return {
        "status": "success",
        "name": name,
        "filename": f"{name}{_MAP_SUFFIX}",
        "map_root": _MAP_ROOT_SPEC,
        "listed_suffixes": list(_LISTED_SUFFIXES),
    }
