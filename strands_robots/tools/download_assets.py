"""Download robot model assets - Strands Agent ``@tool`` wrapper.

Thin wrapper around :mod:`strands_robots.assets.download` that exposes
``download_robots()`` as an agent tool.  All download logic lives in the
``assets.download`` module; this file only handles input parsing and
output formatting for the Strands Agent SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from strands.tools.decorator import tool

from strands_robots.assets.download import download_robots, get_user_assets_dir
from strands_robots.assets.manager import list_available_robots
from strands_robots.registry import format_robot_table
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)


@tool
def download_assets(
    action: str = "download",
    robots: str | None = None,
    category: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download and manage robot model assets (MJCF XML + meshes).

    Assets are sourced from ``robot_descriptions`` (recommended by MuJoCo
    Menagerie, requires ``pip install strands-robots[sim-mujoco]``).  When
    ``robot_descriptions`` is unavailable, falls back to a shallow
    ``git clone`` of the Menagerie repo.  Robots with a custom GitHub
    source in the registry are cloned from their respective repos.

    Downloaded assets are cached in ``~/.strands_robots/assets/``
    (override with ``STRANDS_ASSETS_DIR``).

    Args:
        action: ``download`` | ``list`` | ``status``.  ``status`` marks each
            robot ``[ok]`` (assets present) or ``[--]`` (missing). A ``download``
            that fetched nothing is reported as ``status="error"`` naming the
            cause - an unknown name, a failed clone, or a selection that matched
            no robot - rather than as a success reporting three zeros.
        robots: Comma-separated names (e.g. ``so100,panda``). Omit for all. A
            non-empty value that names no robot (``","``) is refused rather
            than read as "all".
        category: Filter: arm, bimanual, hand, humanoid, mobile, mobile_manip
        force: Re-fetch a robot whose assets are already present, replacing
            the cached directory. A posture, so a non-boolean is refused
            rather than read by truthiness - ``force="false"`` would
            otherwise select the re-fetch it spells the skipping of.
    """
    try:
        if action == "list":
            return {
                "status": "success",
                "content": [{"text": f"Available Robots:\n\n{format_robot_table()}"}],
            }

        if action == "status":
            robots_info = list_available_robots()
            available = sum(1 for r in robots_info if r["available"])
            lines = [f"{available} available, {len(robots_info) - available} missing"]
            lines.extend(
                f"{'[ok]' if r['available'] else '[--]'} {r['name']:<20s} {r['category']:<12s} {r['description']}"
                for r in robots_info
            )
            lines.append(f"\nCache: {get_user_assets_dir()}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        if action == "download":
            # ``robots`` carries a SUBSET of the sim robots as one comma-separated
            # string, and the parse below drops blank fields - so a non-empty
            # argument can name nothing: ``","``, ``" "`` and ``",,,"`` each parse
            # to zero names. Handed on as ``names=[]`` that was the opposite of what
            # it asked for, because ``download_robots`` read the selector by
            # truthiness and downloaded every sim robot instead.
            #
            # ``download_robots`` now refuses an empty selection, so this only has to
            # not manufacture one - but the refusal is raised in terms of ``names=``,
            # which is not the argument this caller passed. Refuse here instead, in
            # this surface's own vocabulary, so the remedy names ``robots=`` and is
            # actionable as written. An absent or empty ``robots`` still means "all":
            # for a single string argument, unset and empty genuinely coincide, and
            # only a value that carries content while naming nothing is a mistake.
            robot_names: list[str] | None = None
            if robots:
                robot_names = [r.strip() for r in robots.split(",") if r.strip()]
                if not robot_names:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"download_assets: robots={robots!r} names no robot - it is read as a "
                                    "comma-separated list and every field in it is blank. Omit robots= to "
                                    'download every sim robot, or name the subset (robots="so100,panda").'
                                )
                            }
                        ],
                    }
            # The facade checks the flag it forwards. ``download_robots`` refuses a
            # non-boolean too, but reached through this surface that ValueError is
            # caught by the handler below and rendered as "Error: ..." with the
            # library function's name in it - a refusal an agent reads as a crash in
            # a function it did not call. Checked here, the caller gets the parameter
            # it passed and the domain it must satisfy. Scoped to this action: the
            # ``list`` and ``status`` actions consume no flag, and an option no
            # handler reads must not be refused.
            if text := boolean_flag_error(force, "force", "download_assets"):
                return {"status": "error", "content": [{"text": text}]}

            result = download_robots(names=robot_names, category=category, force=force)
            # A call that fetched nothing is not a success, and this surface
            # reported one for every way of fetching nothing: "Downloaded: 0,
            # Skipped: 0, Failed: 0" under status="success", which is what a plain
            # typo produces. Three ways in, all through this one call - a name the
            # registry does not list, a clone that failed, and a selection that
            # matched no robot at all.
            #
            # The first two the library now grades for us. The third is
            # ``downloaded + skipped + failed == 0``, and no other outcome can
            # produce it: a selection that is entirely present reports ``skipped``,
            # one that tried and lost reports ``failed``. It is the verdict for an
            # unmatched ``category=`` - the parameter that has no name to report -
            # and the same zeros an all-unknown ``robots=`` arrives with.
            #
            # ``unknown_names`` is still read on its own rather than folded into
            # that count, because the two do not coincide: a selection naming one
            # real robot and one typo has ``skipped == 1``, so only the named list
            # can refuse it.
            unknown: list[str] = list(result.get("unknown_names") or [])
            nothing_selected = not (result["downloaded"] or result["skipped"] or result["failed"])
            parts = [
                f"Downloaded: {result['downloaded']}, Skipped: {result['skipped']}, Failed: {result['failed']}",
            ]
            if unknown:
                parts.append(
                    f"Unknown: {', '.join(unknown)} - not in the registry. "
                    'download_assets(action="list") names every robot it can fetch.'
                )
            if result.get("failed_details"):
                parts.extend(f"   {n}: {r}" for n, r in result["failed_details"].items())
            # A call that never reached a download carries no ``method`` or
            # ``assets_dir``, and rendering them anyway printed "Method: ?" and
            # "Assets: ?" - two placeholders standing in for the reason, under a
            # verdict the caller then had nothing to act on. Every such result
            # carries the library's own ``message`` saying what happened; render
            # that in their place.
            if "method" in result:
                parts.append(f"Method: {result['method']}")
                parts.append(f"Assets: {result.get('assets_dir', '')}")
            elif result.get("message"):
                parts.append(str(result["message"]))
            status = "error" if unknown or result["failed"] or nothing_selected else "success"
            return {"status": status, "content": [{"text": "\n".join(parts)}]}

        return {
            "status": "error",
            "content": [{"text": f"Unknown action: {action}. Valid: download, list, status"}],
        }

    except Exception as exc:
        logger.error("download_assets error: %s", exc)
        return {"status": "error", "content": [{"text": f"Error: {exc}"}]}
