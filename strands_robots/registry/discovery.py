"""Auto-discovery of robots from the optional ``robot_descriptions`` package.

The curated registry (``robots.json``) deliberately carries only robots that
need project-specific metadata: hardware ports, custom joint counts, scene
tweaks, aliases, or local mesh overrides. The much larger long tail of standard
robots shipped by ``robot_descriptions`` (MuJoCo Menagerie and friends) does not
need a hand-written entry - this module resolves those on demand so
``Robot("go2", mode="sim")`` works without touching ``robots.json``.

Resolution rules:
    - A curated ``robots.json`` entry always wins. Discovery is consulted only
      for names unknown to the curated registry (see
      :func:`strands_robots.registry.get_robot`).
    - Curated-registry / MJCF discovery (``descriptions_module``,
      ``list_discoverable``, ``discover_robot``) is MJCF-only: the MuJoCo backend
      and the curated registry need an ``.xml`` model.
    - URDF discovery (``urdf_descriptions_module``, ``list_urdf_discoverable``,
      ``discover_urdf_path``) is a parallel surface for URDF-native backends
      (Newton via ``ModelBuilder.add_urdf``). It covers the large URDF-only long
      tail that MJCF discovery cannot (humanoids, quadrupeds, hands).
    - The MuJoCo backend reaches that same long tail through
      :func:`discover_robot`: a name with a URDF description and no MJCF one gets
      an entry whose ``asset.source.type`` is ``"urdf"``, and
      :mod:`strands_robots.assets.urdf` compiles the URDF into ``robot.xml`` +
      ``scene.xml`` on first resolution (:func:`list_urdf_only` names them).
    - :func:`descriptions_module`, :func:`is_discoverable`, and
      :func:`list_discoverable` are cheap - a static dict lookup with no module
      import and no network. :func:`discover_robot` is heavy: importing a
      description module makes ``robot_descriptions`` clone the upstream asset
      repository on first use, so it is called only from asset-resolution paths
      that are already allowed to download.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from .._description_cache import import_description
from ..utils import log_safe
from .loader import _load, normalize_robot_name

logger = logging.getLogger(__name__)

# Robot names are interpolated into ``importlib`` module paths, so restrict them
# to a conservative allowlist before any lookup - no dots, slashes, or
# whitespace can reach the import machinery.
_NAME_RE = re.compile(r"^[a-z0-9_]+\Z")

# ``robot_descriptions`` names every MuJoCo (MJCF) description module with this
# suffix, e.g. ``go2_mj_description`` -> canonical robot name ``go2``.
_MJCF_SUFFIX = "_mj_description"

# URDF description modules use the bare ``_description`` suffix (e.g.
# ``panda_description``). MJCF modules use ``_mj_description``; the URDF table
# below excludes those. URDF descriptions are consumable only by URDF-native
# backends (Newton via ``ModelBuilder.add_urdf``); the MuJoCo backend ignores
# them because it needs an MJCF ``.xml`` model.
_URDF_SUFFIX = "_description"

# Candidate scene files (ground plane + lights) a Menagerie description may ship
# alongside the bare robot model, in preference order.
_SCENE_CANDIDATES = ("scene.xml", "scene_mjx.xml")

# Cache of synthesized entries (and negative results) keyed by normalized name.
_DISCOVER_CACHE: dict[str, dict[str, Any] | None] = {}


@lru_cache(maxsize=1)
def _mjcf_modules() -> dict[str, str]:
    """Map canonical robot name -> MJCF description module name.

    Reads ``robot_descriptions``' static description table, which is a plain
    dict - no module import and no network. Returns an empty mapping when
    ``robot_descriptions`` is not installed.
    """
    try:
        from robot_descriptions._descriptions import (  # type: ignore[import-not-found]
            DESCRIPTIONS,
            Format,
        )
    except ImportError:
        return {}

    mapping: dict[str, str] = {}
    for module_name, desc in DESCRIPTIONS.items():
        if Format.MJCF not in desc.formats:
            continue
        if not module_name.endswith(_MJCF_SUFFIX):
            continue
        short = module_name[: -len(_MJCF_SUFFIX)]
        mapping[short] = module_name
    return mapping


def descriptions_module(name: str) -> str | None:
    """Return the MJCF ``robot_descriptions`` module for *name*, or ``None``.

    Cheap: a dict lookup against the static description table, with no module
    import and no network. Returns ``None`` when the robot is not an
    MJCF-capable ``robot_descriptions`` robot or when the package is missing.

    Examples::

        descriptions_module("go2")   # -> "go2_mj_description"
        descriptions_module("so100") # -> None (curated, not a description)
    """
    norm = normalize_robot_name(name)
    if not _NAME_RE.match(norm):
        return None
    return _mjcf_modules().get(norm)


def is_discoverable(name: str) -> bool:
    """Return ``True`` if *name* resolves from ``robot_descriptions`` (cheap)."""
    return descriptions_module(name) is not None


def list_discoverable() -> list[str]:
    """Return sorted canonical names resolvable from ``robot_descriptions``.

    This is the MJCF long tail - the standard robots that work in the MuJoCo
    backend without a curated ``robots.json`` entry (e.g. ``go2``, ``spot``,
    ``h1``, ``anymal_c``, ``cassie``). Cheap: no import, no network.
    """
    return sorted(_mjcf_modules())


def discover_robot(name: str) -> dict[str, Any] | None:
    """Synthesize a registry entry for *name* from ``robot_descriptions``.

    Heavy: imports the description module, which makes ``robot_descriptions``
    clone the upstream asset repository on first use. Call only from
    asset-resolution paths that are allowed to download. Results (including
    misses) are cached.

    Args:
        name: Robot name or alias (e.g. ``"go2"``).

    Returns:
        A registry-style entry with the same shape as a ``robots.json`` value -
        an ``asset`` block wired to the resolved ``robot_descriptions`` module -
        or ``None`` if the robot is not an MJCF-capable ``robot_descriptions``
        robot.
    """
    norm = normalize_robot_name(name)
    if norm in _DISCOVER_CACHE:
        return _DISCOVER_CACHE[norm]

    module_name = descriptions_module(norm)
    if module_name is None:
        entry_urdf = _urdf_entry(norm)
        _DISCOVER_CACHE[norm] = entry_urdf
        return entry_urdf

    try:
        mod = import_description(module_name)
    except ImportError as exc:
        logger.debug("Discovery import failed for %s (%s): %s", log_safe(norm), module_name, exc)
        _DISCOVER_CACHE[norm] = None
        return None

    mjcf_path = getattr(mod, "MJCF_PATH", None)
    package_path = getattr(mod, "PACKAGE_PATH", None)
    if not mjcf_path or not package_path:
        logger.warning(
            "robot_descriptions module %r lacks MJCF_PATH/PACKAGE_PATH; cannot discover %s",
            module_name,
            log_safe(norm),
        )
        _DISCOVER_CACHE[norm] = None
        return None

    package_dir = str(package_path)
    asset_dir = os.path.basename(os.path.normpath(package_dir))
    model_xml = os.path.basename(str(mjcf_path))

    # Prefer a scene file (ground + lights) when the description ships one;
    # otherwise fall back to the bare model so rendering still has something.
    scene_xml = model_xml
    for candidate in _SCENE_CANDIDATES:
        if os.path.exists(os.path.join(package_dir, candidate)):
            scene_xml = candidate
            break

    entry: dict[str, Any] = {
        "description": f"{norm} (discovered via robot_descriptions:{module_name})",
        "category": "discovered",
        "discovered": True,
        "asset": {
            "dir": asset_dir,
            "model_xml": model_xml,
            "scene_xml": scene_xml,
            "robot_descriptions_module": module_name,
        },
    }
    _DISCOVER_CACHE[norm] = entry
    return entry


@lru_cache(maxsize=1)
def _urdf_modules() -> dict[str, str]:
    """Map canonical robot name -> URDF ``robot_descriptions`` module name.

    The complement of :func:`_mjcf_modules`: every description that ships a URDF
    model (and is not the MJCF ``_mj_description`` variant). URDF-native backends
    such as Newton can ingest these directly via ``ModelBuilder.add_urdf`` even
    when no MJCF model exists, which unlocks the large URDF-only long tail
    (humanoids, quadrupeds, hands) absent from MJCF discovery. Reads
    ``robot_descriptions``' static description table - no module import and no
    network. Returns an empty mapping when ``robot_descriptions`` is not
    installed.
    """
    try:
        from robot_descriptions._descriptions import (  # type: ignore[import-not-found]
            DESCRIPTIONS,
            Format,
        )
    except ImportError:
        return {}

    mapping: dict[str, str] = {}
    for module_name, desc in DESCRIPTIONS.items():
        if Format.URDF not in desc.formats:
            continue
        if not module_name.endswith(_URDF_SUFFIX):
            continue
        # ``_mj_description`` also ends with ``_description`` but is the MJCF
        # variant; it is handled by :func:`_mjcf_modules`, not here.
        if module_name.endswith(_MJCF_SUFFIX):
            continue
        short = module_name[: -len(_URDF_SUFFIX)]
        mapping[short] = module_name
    return mapping


def urdf_descriptions_module(name: str) -> str | None:
    """Return the URDF ``robot_descriptions`` module for *name*, or ``None``.

    Cheap: a dict lookup against the static description table, with no module
    import and no network. Returns ``None`` when *name* is not a URDF-capable
    ``robot_descriptions`` robot or when the package is missing.

    Examples::

        urdf_descriptions_module("panda")    # -> "panda_description"
        urdf_descriptions_module("atlas_v4") # -> "atlas_v4_description"
        urdf_descriptions_module("so100")    # -> None (curated, not a description)
    """
    norm = normalize_robot_name(name)
    if not _NAME_RE.match(norm):
        return None
    return _urdf_modules().get(norm)


def is_urdf_discoverable(name: str) -> bool:
    """Return ``True`` if *name* resolves to a URDF ``robot_descriptions`` model."""
    return urdf_descriptions_module(name) is not None


def list_urdf_discoverable() -> list[str]:
    """Return sorted canonical names resolvable to a URDF from ``robot_descriptions``.

    This is the URDF long tail consumable by URDF-native backends (Newton). It
    includes robots with no MJCF model at all (e.g. ``atlas_v4``, ``baxter``,
    ``b1``), which is why it is disjoint from :func:`list_discoverable` for those
    entries. Names the sweep recorded with ``has_sim: false`` (upstream clone
    failed or ``URDF_PATH`` missing after import; see ``urdf_robots.json``) are
    omitted so a caller cannot land on a dead end from this list. Cheap: no
    import, no network.
    """
    broken = _urdf_unbuildable_set()
    return sorted(n for n in _urdf_modules() if n not in broken)


def discover_urdf_path(name: str) -> str | None:
    """Resolve the on-disk URDF path for *name* via ``robot_descriptions``.

    Heavy: imports the description module, which makes ``robot_descriptions``
    clone the upstream asset repository on first use. Call only from
    asset-resolution paths that are allowed to download.

    Args:
        name: Robot name or alias (e.g. ``"panda"``).

    Returns:
        Absolute path to the URDF file, or ``None`` when *name* is not a
        URDF-capable ``robot_descriptions`` robot, the module cannot be
        imported, or it exposes no readable ``URDF_PATH``.
    """
    norm = normalize_robot_name(name)
    module_name = urdf_descriptions_module(norm)
    if module_name is None:
        return None

    try:
        mod = import_description(module_name)
    except ImportError as exc:
        logger.debug("URDF discovery import failed for %r (%s): %s", norm, module_name, exc)
        return None

    urdf_path = getattr(mod, "URDF_PATH", None)
    if not urdf_path or not os.path.exists(str(urdf_path)):
        logger.warning(
            "robot_descriptions module %r lacks a readable URDF_PATH; cannot discover %r",
            module_name,
            norm,
        )
        return None
    return str(urdf_path)


#: Marker the URDF entries carry in ``asset.source`` so the downloader builds
#: instead of symlinking, and so listings can say where a robot came from.
URDF_SOURCE_TYPE = "urdf"


def urdf_registry() -> dict[str, dict[str, Any]]:
    """The sweep-generated ``urdf_robots.json`` table, keyed by canonical name.

    Written by ``scripts/build_urdf_registry.py`` from assets the loader built:
    category, actuated joint count (``joints`` = the compiled model's ``nu``),
    floating base, upstream pin, mesh formats, ``has_sim`` and the refusal for
    a description that does not build. Cheap: a JSON read, cached by the
    registry loader.
    """
    table = _load("urdf_robots").get("robots", {})
    return dict(table) if isinstance(table, dict) else {}


def _urdf_entry(norm: str) -> dict[str, Any] | None:
    """Registry-style entry for a URDF-only ``robot_descriptions`` robot, or ``None``.

    Cheap: static tables and :func:`urdf_registry` only. The asset directory is
    the module name (unique, self-describing, never a Menagerie directory); the
    model and scene files are the ones :mod:`strands_robots.assets.urdf` writes
    there on first resolution. ``category``, ``joints`` and ``floating`` come
    from the sweep table when the robot is in it, else from the description's
    tags through the same mapping the builder uses, so the docs family and the
    sim agree. A robot the sweep could not build has no ``asset`` block and
    carries the loader's ``refusal`` instead, so ``has_sim`` says no.
    """
    module_name = urdf_descriptions_module(norm)
    if module_name is None:
        return None
    from ..assets.urdf import ASSET_JSON, ROBOT_XML, SCENE_XML, category_for_tags, is_floating

    recorded = urdf_registry().get(norm, {})
    tags: set[str] = set(recorded.get("tags", ()))
    if not tags:
        try:
            from robot_descriptions._descriptions import DESCRIPTIONS  # type: ignore[import-not-found]

            tags = set(getattr(DESCRIPTIONS.get(module_name), "tags", ()) or ())
        except ImportError:  # pragma: no cover - urdf_descriptions_module already needed the package
            pass
    entry: dict[str, Any] = {
        "description": f"{norm} ({' '.join(sorted(tags)) or 'robot'} URDF from robot_descriptions)",
        "category": recorded.get("category") or category_for_tags(tags),
        "discovered": True,
        "source": URDF_SOURCE_TYPE,
        "floating": bool(recorded.get("floating", is_floating(tags))),
    }
    if recorded.get("joints") is not None:
        entry["joints"] = int(recorded["joints"])
    if recorded.get("has_sim", True):
        entry["asset"] = {
            "dir": module_name,
            "model_xml": ROBOT_XML,
            "scene_xml": SCENE_XML,
            "info_json": ASSET_JSON,
            "robot_descriptions_module": module_name,
            "source": {"type": URDF_SOURCE_TYPE},
        }
    else:
        entry["refusal"] = str(recorded.get("refusal", "the description does not build"))
    return entry


def urdf_registry_entry(name: str) -> dict[str, Any] | None:
    """The registry entry for a URDF-only robot by name or ``None``; cheap, no import."""
    norm = normalize_robot_name(name)
    if not _NAME_RE.match(norm) or not is_urdf_only(norm):
        return None
    return _urdf_entry(norm)


def list_urdf_only() -> list[str]:
    """Sorted URDF names the MuJoCo backend gains: no MJCF description, not curated.

    The complement of :func:`list_discoverable` inside :func:`list_urdf_discoverable`,
    minus every curated name and alias (curated always wins, and an alias must
    stay unique). Names the sweep recorded with ``has_sim: false`` are omitted
    for the same reason as :func:`list_urdf_discoverable`: the robot's
    description does not compile, so the backend does not actually gain it.
    Membership (:func:`is_urdf_only`) still returns ``True`` for those names so
    the "URDF that does not compile" refusal path can still fire from
    :mod:`strands_robots.simulation.base`. Cheap: static tables and the curated
    registry, no import.
    """
    broken = _urdf_unbuildable_set()
    return sorted(n for n in _urdf_only_set() if n not in broken)


def is_urdf_only(name: str) -> bool:
    """Return ``True`` if *name* is a robot the MuJoCo backend serves from a URDF.

    The membership test is for every URDF description that is not curated and
    has no MJCF model, whether or not the sweep could build it: a name that
    did not build still routes to :func:`urdf_registry_entry`, which carries
    the sweep's ``refusal`` and lets
    :mod:`strands_robots.simulation.base` render the "URDF that does not
    compile" error. :func:`list_urdf_only` is the subset that *did* build and
    is safe to advertise. Cheap: no import, no network.
    """
    norm = normalize_robot_name(name)
    return _NAME_RE.match(norm) is not None and norm in _urdf_only_set()


@lru_cache(maxsize=1)
def _urdf_only_set() -> frozenset[str]:
    """Every URDF description that is not curated and has no MJCF model.

    Includes names the sweep marked ``has_sim: false`` so :func:`is_urdf_only`
    (and :func:`urdf_registry_entry` through it) can route them to the correct
    refusal. The user-facing listings filter the broken subset out through
    :func:`_urdf_unbuildable_set`.
    """
    reg = _load("robots").get("robots", {})
    taken = set(reg)
    for info in reg.values():
        taken.update(info.get("aliases", ()))
    mjcf = _mjcf_modules()
    return frozenset(n for n in _urdf_modules() if n not in mjcf and n not in taken)


@lru_cache(maxsize=1)
def _urdf_unbuildable_set() -> frozenset[str]:
    """Names the sweep marked ``has_sim: false`` in ``urdf_robots.json``.

    These are description modules the builder could not compile at sweep time
    (``urdf_robots.json`` records the reason in ``refusal``). The lists keep
    them out so a caller never sees a name they cannot ``Robot(...)``. Cheap:
    one JSON table read, cached for the life of the process.
    """
    table = _load("urdf_robots").get("robots", {})
    if not isinstance(table, dict):
        return frozenset()
    return frozenset(
        name for name, info in table.items() if isinstance(info, dict) and info.get("has_sim") is False
    )


def invalidate_cache() -> None:
    """Clear the discovery caches (synthesized entries + module table)."""
    _DISCOVER_CACHE.clear()
    # ``_mjcf_modules`` / ``_urdf_modules`` are normally ``lru_cache``-wrapped;
    # guard ``cache_clear`` so a test (or future refactor) that swaps in a plain
    # callable cannot break this.
    for cached in (_mjcf_modules, _urdf_modules, _urdf_only_set, _urdf_unbuildable_set):
        clear = getattr(cached, "cache_clear", None)
        if clear is not None:
            clear()
