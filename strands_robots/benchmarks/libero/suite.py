"""Bulk benchmark registration for LIBERO task suites.

LIBERO ships ~130 tasks split across five suites:

* ``libero-spatial`` - 10 tasks, same objects / different spatial goals
* ``libero-object`` - 10 tasks, different objects / same goal structure
* ``libero-goal`` - 10 tasks, same objects / different goals
* ``libero-10`` - 10 tasks, "short-horizon diverse"
* ``libero-90`` - 90 tasks, "long-horizon diverse"

Rather than have agents call :func:`register_benchmark` 130× manually,
:func:`load_libero_suite` walks the upstream package's BDDL directory and
registers every task under a predictable ``libero-<suite>-<task>`` key.
Tasks that fail to parse are logged and skipped - a single malformed BDDL
file should never block the whole suite from loading.

Layout discovery
----------------

The ``libero`` pip package historically keeps BDDL files under
``<libero_root>/libero/bddl_files/<suite_name>/*.bddl`` and scene MJCFs
alongside the benchmark code. Because the exact subpath has drifted
between releases, :func:`load_libero_suite` accepts an explicit
``bddl_dir=`` override and falls back to probing a handful of standard
locations when not given. The scene resolver behaves similarly via
``scene_dir=``.

Callers who already have the BDDL files on disk (e.g. vendored into their
repo) do **not** need the ``libero`` package installed - just pass
``bddl_dir=`` and the function registers from there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from strands_robots.benchmarks.libero.adapter import LiberoAdapter
from strands_robots.benchmarks.libero.bddl_parser import BDDLParseError
from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.utils import require_optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Canonical suite names - these are the keys agents will use. The upstream
# directory names sometimes differ (snake_case vs kebab-case); the name
# resolver accepts both.
SUITE_NAMES = frozenset(
    {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    }
)


def _normalise_suite_name(name: str) -> str:
    """Accept either ``libero-spatial`` or ``libero_spatial``; return the underscore form.

    Upstream LIBERO uses snake_case directory names; our benchmark registry
    uses kebab-case keys. Keep the normalisation centralised.
    """
    key = name.strip().lower().replace("-", "_")
    if not key.startswith("libero_"):
        key = f"libero_{key}"
    return key


def _candidate_bddl_dirs(libero_root: Path, suite: str) -> list[Path]:
    """Return paths to try in order. First existing one wins."""
    return [
        libero_root / "libero" / "bddl_files" / suite,
        libero_root / "libero" / "libero" / "bddl_files" / suite,
        libero_root / "bddl_files" / suite,
        libero_root / "libero" / "tasks" / suite,
        libero_root / suite,
    ]


def _resolve_libero_root() -> Path:
    """Find the filesystem root of the installed ``libero`` package.

    Lazily imports ``libero`` via :func:`require_optional` with a helpful
    install hint pointing at ``strands-robots[benchmark-libero]``.
    """
    libero = require_optional(
        "libero",
        pip_install="libero",
        extra="benchmark-libero",
        purpose="LIBERO benchmark suite discovery",
    )
    # __file__ lives inside the package; its parent is the package root.
    libero_file = getattr(libero, "__file__", None)
    if not libero_file:
        raise RuntimeError("libero package has no __file__ attribute; cannot locate BDDL tasks")
    return Path(libero_file).resolve().parent.parent


def load_libero_suite(
    suite_name: str,
    *,
    bddl_dir: str | Path | None = None,
    scene_dir: str | Path | None = None,
    max_steps: int | None = None,
    init_jitter: float = 0.02,
    key_prefix: str = "libero",
) -> dict[str, LiberoAdapter]:
    """Register every task in ``suite_name`` under the benchmark registry.

    Args:
        suite_name: One of ``libero_spatial`` / ``libero_object`` /
            ``libero_goal`` / ``libero_10`` / ``libero_90``. Accepts
            ``libero-spatial`` form too.
        bddl_dir: Explicit directory containing ``*.bddl`` files. When
            omitted, tries the installed ``libero`` package layout.
        scene_dir: Root under which per-task scene MJCFs live. When
            provided, each adapter gets ``scene_path = scene_dir /
            <task>.xml`` if the file exists; otherwise scene is left as
            ``None`` and the adapter assumes the scene is already loaded.
        max_steps: Forwarded to every :class:`LiberoAdapter`.
        init_jitter: Forwarded to every :class:`LiberoAdapter`.
        key_prefix: Registry key format is ``<key_prefix>-<suite>-<task>``.
            Pass ``key_prefix=""`` for ``<suite>-<task>``.

    Returns:
        ``{registry_name: LiberoAdapter}`` for every successfully registered
        task. Failed tasks are logged (at WARNING) and omitted.

    Raises:
        FileNotFoundError: If no BDDL directory can be located.
        ValueError: If ``suite_name`` isn't a recognised LIBERO suite.
    """
    suite = _normalise_suite_name(suite_name)
    if suite not in SUITE_NAMES:
        raise ValueError(f"Unknown LIBERO suite {suite_name!r}. Valid: {sorted(SUITE_NAMES)}")

    resolved_bddl_dir = _locate_bddl_dir(suite, bddl_dir)
    resolved_scene_dir = Path(scene_dir).expanduser().resolve() if scene_dir else None

    registered: dict[str, LiberoAdapter] = {}
    failures: list[tuple[str, str]] = []

    for bddl_file in sorted(resolved_bddl_dir.glob("*.bddl")):
        task_stem = bddl_file.stem
        registry_name = _format_registry_name(key_prefix, suite, task_stem)

        scene_path: str | None = None
        if resolved_scene_dir is not None:
            candidate = resolved_scene_dir / f"{task_stem}.xml"
            if candidate.exists():
                scene_path = str(candidate)

        try:
            adapter = LiberoAdapter.from_file(
                bddl_file,
                scene_path=scene_path,
                max_steps=max_steps,
                init_jitter=init_jitter,
            )
        except (BDDLParseError, FileNotFoundError, ValueError) as e:
            logger.warning("Skipping LIBERO task %s: %s", bddl_file.name, e)
            failures.append((str(bddl_file), str(e)))
            continue
        register_benchmark(registry_name, adapter)
        registered[registry_name] = adapter

    logger.info(
        "📚 Registered %d LIBERO tasks from %s (skipped %d malformed)",
        len(registered),
        resolved_bddl_dir,
        len(failures),
    )
    return registered


def _format_registry_name(prefix: str, suite: str, task: str) -> str:
    # Suite is in ``libero_spatial`` form; keys use kebab-case.
    # * With prefix: ``<prefix>-<suite-without-libero>-<task>``
    #   (``libero-spatial-pick_cube`` when prefix="libero").
    # * Without prefix: ``<suite-without-libero>-<task>``
    #   (``spatial-pick_cube``) - agents who supply their own key scheme
    #   don't want the ``libero-`` doubled in.
    suite_kebab = suite.replace("_", "-").removeprefix("libero-")
    if prefix:
        return f"{prefix}-{suite_kebab}-{task}"
    return f"{suite_kebab}-{task}"


def _locate_bddl_dir(suite: str, override: str | Path | None) -> Path:
    if override is not None:
        d = Path(override).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"bddl_dir does not exist or is not a directory: {d}")
        return d

    libero_root = _resolve_libero_root()
    for candidate in _candidate_bddl_dirs(libero_root, suite):
        if candidate.is_dir():
            return candidate
    tried = [str(p) for p in _candidate_bddl_dirs(libero_root, suite)]
    raise FileNotFoundError(
        f"Could not locate BDDL directory for suite {suite!r}. Tried: {tried}. Pass bddl_dir= to override."
    )


def available_suites() -> Iterable[str]:
    """Return the canonical set of LIBERO suite names - offline constant."""
    return frozenset(SUITE_NAMES)


__all__ = [
    "SUITE_NAMES",
    "available_suites",
    "load_libero_suite",
]
