"""Download robot model assets via ``robot_descriptions`` or custom GitHub repos.

The recommended way to obtain MuJoCo Menagerie models is the
`robot_descriptions <https://pypi.org/project/robot_descriptions/>`_ package
(see `Menagerie README — Via robot-descriptions
<https://github.com/google-deepmind/mujoco_menagerie#via-robot-descriptions>`_).

``robot_descriptions`` clones the Menagerie repo once into
``~/.cache/robot_descriptions/mujoco_menagerie/`` and exposes per-robot
``PACKAGE_PATH`` (directory with meshes + XML) and ``MJCF_PATH`` attributes.

This module:
    1. **Auto-discovers** which ``robot_descriptions`` modules map to which
       Menagerie directories — no hardcoded list needed.
    2. Uses ``robot_descriptions`` for all Menagerie robots (primary path).
    3. Falls back to a shallow ``git clone`` when ``robot_descriptions`` is
       not installed or the robot is not in Menagerie.
    4. Downloads from custom GitHub repos for robots with
       ``asset.source`` in ``registry/robots.json``.

Assets are symlinked/copied into ``~/.strands_robots/assets/`` so the
unified search path in ``assets/__init__.py`` finds them transparently.

Install the optional dependency::

    pip install strands-robots[sim]   # includes robot_descriptions

Works as both a CLI tool and a Strands AgentTool:

CLI::

    python -m strands_robots.tools.download_assets
    python -m strands_robots.tools.download_assets so100 panda unitree_g1
    python -m strands_robots.tools.download_assets --category arm
    python -m strands_robots.tools.download_assets --list

Agent::

    from strands_robots.tools.download_assets import download_assets
    agent = Agent(tools=[download_assets])
    agent("Download the SO-100 and Panda robot assets")
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from strands.tools.decorator import tool
except ImportError:

    def tool(f):
        return f


from strands_robots.assets import format_robot_table, get_search_paths
from strands_robots.registry import get_robot
from strands_robots.registry import list_robots as registry_list_robots
from strands_robots.registry import resolve_name as resolve_robot_name

logger = logging.getLogger(__name__)

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"


# ─────────────────────────────────────────────────────────────────────
# Auto-discover robot_descriptions → menagerie dir mapping
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _discover_rd_mapping() -> Dict[str, str]:
    """Auto-discover the menagerie_dir → robot_descriptions module mapping.

    Scans all ``*_mj_description.py`` files in the ``robot_descriptions``
    package and extracts the Menagerie directory name from the
    ``PACKAGE_PATH`` definition (static analysis — no imports, no git clones).

    Returns:
        Dict mapping menagerie directory name → robot_descriptions module name.
        Empty dict if robot_descriptions is not installed.

    Example::

        >>> _discover_rd_mapping()
        {
            "franka_emika_panda": "panda_mj_description",
            "unitree_g1": "g1_mj_description",
            ...
        }
    """
    try:
        import robot_descriptions

        rd_dir = os.path.dirname(robot_descriptions.__file__)
    except ImportError:
        return {}

    mapping: Dict[str, str] = {}

    for fname in os.listdir(rd_dir):
        if not fname.endswith("_mj_description.py"):
            continue

        modname = fname[:-3]  # strip .py
        fpath = os.path.join(rd_dir, fname)

        try:
            with open(fpath, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue

        # Extract the Menagerie directory from PACKAGE_PATH.
        #
        # Standard Menagerie modules use a single-arg join:
        #   PACKAGE_PATH = _path.join(REPOSITORY_PATH, "unitree_g1")
        #
        # Non-Menagerie repos use multi-arg joins (we skip these):
        #   PACKAGE_PATH = _path.join(REPOSITORY_PATH, "data", "a1")
        #   PACKAGE_PATH = _path.join(REPOSITORY_PATH, "Simulation", "SO101")
        #
        # The regex requires ) right after the closing quote so that
        # multi-arg joins don't match.
        pkg_match = re.search(
            r'PACKAGE_PATH\s*(?::.*?)?=\s*_path\.join\(\s*REPOSITORY_PATH\s*,'
            r'\s*["\']([^"\']+)["\']\s*\)',
            src,
        )
        if pkg_match:
            menagerie_dir = pkg_match.group(1)
            mapping[menagerie_dir] = modname

    logger.debug(
        "Auto-discovered %d robot_descriptions MJ modules", len(mapping)
    )
    return mapping


# ─────────────────────────────────────────────────────────────────────
# User cache directory (not bundled in pip package)
# ─────────────────────────────────────────────────────────────────────


def get_user_assets_dir() -> Path:
    """Get user-level asset cache directory."""
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom)
    else:
        d = Path.home() / ".strands_robots" / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _robot_descriptions_available() -> bool:
    """Check if robot_descriptions is installed."""
    try:
        import robot_descriptions  # noqa: F401

        return True
    except ImportError:
        return False


def _needs_download(name: str, info: dict, force: bool = False) -> bool:
    """Check if a robot needs downloading (mesh files missing)."""
    asset = info.get("asset", {})
    if not asset:
        return False

    xml_file = asset["model_xml"]
    asset_dir = asset["dir"]

    for search_dir in get_search_paths():
        model_path = search_dir / asset_dir / xml_file
        if not model_path.exists():
            continue
        # XML exists — check mesh files
        try:
            content = model_path.read_text()
            mesh_files = re.findall(
                r'file="([^"]+\.(?:stl|STL|obj|OBJ|msh))"', content
            )
            if not mesh_files:
                return False  # No meshes needed
            meshdir_match = re.search(r'meshdir="([^"]*)"', content)
            meshdir = meshdir_match.group(1) if meshdir_match else ""
            for mf in mesh_files[:3]:
                mesh_path = model_path.parent / meshdir / mf
                if not mesh_path.exists():
                    return True  # Missing mesh
            return force  # All meshes present
        except Exception:
            return True  # Can't read XML, re-download

    return True  # XML not found anywhere


def _get_source(info: dict) -> dict:
    """Get download source for a robot.  Defaults to menagerie."""
    asset = info.get("asset", {})
    source = asset.get("source", {})
    if source:
        return source
    # Default: MuJoCo Menagerie
    return {"type": "menagerie"}


def _safe_join(base: Path, untrusted: str) -> Path:
    """Join *base* with an untrusted relative path, rejecting traversal.

    Uses ``os.path.normpath`` (not ``resolve``) so that existing symlinks
    inside *base* are not followed — we only need to block ``..`` escapes.

    Raises ``ValueError`` if the normalised result escapes *base*.
    """
    # Normalise without following symlinks (resolve() would chase them)
    joined = Path(os.path.normpath(base / untrusted))
    base_norm = Path(os.path.normpath(base))
    if not (joined == base_norm or str(joined).startswith(str(base_norm) + os.sep)):
        raise ValueError(
            f"Path traversal blocked: {untrusted!r} escapes {base}"
        )
    return joined



# ─────────────────────────────────────────────────────────────────────
# Primary path: robot_descriptions
# ─────────────────────────────────────────────────────────────────────


def _download_via_robot_descriptions(
    robots: Dict[str, dict],
    dest_dir: Path,
) -> Dict[str, str]:
    """Download robots using the ``robot_descriptions`` package.

    ``robot_descriptions`` handles cloning / caching Menagerie internally.
    We import the per-robot module to trigger the clone, then symlink or
    copy from its cache into our own ``~/.strands_robots/assets/`` tree.

    Returns:
        Dict mapping robot_name → "downloaded" | "failed: reason".
    """
    results: Dict[str, str] = {}
    if not robots:
        return results

    import importlib

    rd_mapping = _discover_rd_mapping()

    for name, info in robots.items():
        asset_dir = info["asset"]["dir"]

        # Look up from registry override first, then auto-discovered mapping
        rd_module_name = info["asset"].get("rd_module") or rd_mapping.get(asset_dir)

        if rd_module_name is None:
            results[name] = "skipped: no robot_descriptions module found"
            continue

        try:
            # Validate module name to prevent import injection
            if not re.match(r'^[a-z0-9_]+_mj_description$', rd_module_name):
                results[name] = f"skipped: invalid module name: {rd_module_name}"
                continue
            full_module = f"robot_descriptions.{rd_module_name}"
            logger.info(
                "Loading %s via robot_descriptions (%s)...", name, full_module
            )
            mod = importlib.import_module(full_module)
            package_path = Path(mod.PACKAGE_PATH)

            if not package_path.exists():
                results[name] = (
                    f"failed: PACKAGE_PATH does not exist: {package_path}"
                )
                continue

            dst = _safe_join(dest_dir, asset_dir)

            # Symlink if possible (saves disk), copy otherwise
            if dst.exists():
                if dst.is_symlink() and dst.resolve() == package_path.resolve():
                    results[name] = "downloaded"
                    continue
                # Remove stale destination
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(str(dst))

            try:
                dst.symlink_to(package_path)
                logger.info("Symlinked %s → %s", dst, package_path)
            except OSError:
                # Symlink not supported (Windows, cross-device, etc.)
                shutil.copytree(
                    str(package_path), str(dst), dirs_exist_ok=True
                )
                logger.info("Copied %s → %s", package_path, dst)

            results[name] = "downloaded"
            logger.info("Downloaded %s via robot_descriptions", name)

        except Exception as e:
            results[name] = f"failed: {e}"
            logger.warning("robot_descriptions failed for %s: %s", name, e)

    return results


# ─────────────────────────────────────────────────────────────────────
# Fallback: git clone (when robot_descriptions not installed)
# ─────────────────────────────────────────────────────────────────────


def _download_from_menagerie_git(
    robots: Dict[str, dict],
    dest_dir: Path,
) -> Dict[str, str]:
    """Fallback: clone Menagerie via git and copy robot directories.

    Used when ``robot_descriptions`` is not installed.

    Returns:
        Dict mapping robot_name → "downloaded" | "failed: reason".
    """
    results: Dict[str, str] = {}
    if not robots:
        return results

    logger.info(
        "robot_descriptions not available — falling back to git clone for "
        "%d robots...",
        len(robots),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "mujoco_menagerie")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    MENAGERIE_REPO,
                    clone_dir,
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            for name in robots:
                results[name] = "failed: git clone timeout"
            return results
        except subprocess.CalledProcessError as e:
            for name in robots:
                results[name] = (
                    f"failed: git clone error: "
                    f"{e.stderr.decode()[:100]}"
                )
            return results

        for name, info in robots.items():
            asset_dir = info["asset"]["dir"]
            src = _safe_join(Path(clone_dir), asset_dir)
            dst = _safe_join(dest_dir, asset_dir)

            if not src.exists():
                results[name] = f"failed: {asset_dir} not in menagerie"
                continue
            try:
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                # Clean non-essential files
                for pattern in [
                    "README.md",
                    "LICENSE",
                    "*.png",
                    "*.jpg",
                ]:
                    for f in dst.glob(pattern):
                        f.unlink()
                results[name] = "downloaded"
                logger.info(
                    "Downloaded %s from menagerie (git clone)", name
                )
            except Exception as e:
                results[name] = f"failed: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────
# Custom GitHub repos (non-Menagerie robots)
# ─────────────────────────────────────────────────────────────────────


def _download_from_github(
    name: str,
    info: dict,
    dest_dir: Path,
) -> str:
    """Download a robot from a custom GitHub repo.

    Uses the ``asset.source`` config::

        {"type": "github", "repo": "owner/name", "subdir": "path/in/repo"}

    Returns:
        "downloaded" or "failed: reason"
    """
    source = info["asset"]["source"]
    repo = source["repo"]
    subdir = source.get("subdir", "")
    asset_dir = info["asset"]["dir"]

    # Validate repo format to prevent URL injection
    if not re.match(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$', repo):
        return f"failed: invalid repo format: {repo}"
    repo_url = f"https://github.com/{repo}.git"
    logger.info(
        "Downloading %s from %s (subdir: %s)...",
        name,
        repo,
        subdir or "/",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    repo_url,
                    clone_dir,
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "failed: git clone timeout"
        except subprocess.CalledProcessError as e:
            return (
                f"failed: git clone error: {e.stderr.decode()[:100]}"
            )

        src = Path(clone_dir) / subdir if subdir else Path(clone_dir)
        if not src.exists():
            return f"failed: subdir '{subdir}' not found in {repo}"

        dst = _safe_join(dest_dir, asset_dir)
        try:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            # Clean non-essential
            for pattern in [
                "README.md",
                "LICENSE",
                "*.png",
                "*.jpg",
                ".git*",
            ]:
                for f in dst.glob(pattern):
                    if f.is_file():
                        f.unlink()

            # Copy bundled XML files from package so mesh paths resolve
            bundled_dir = (
                Path(__file__).parent.parent / "assets" / asset_dir
            )
            if bundled_dir.exists():
                for xml_file in bundled_dir.glob("**/*.xml"):
                    rel = xml_file.relative_to(bundled_dir)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(str(xml_file), str(target))
                        logger.debug("Copied bundled XML: %s", rel)

            logger.info("Downloaded %s from %s", name, repo)
            return "downloaded"
        except Exception as e:
            return f"failed: {e}"


# ─────────────────────────────────────────────────────────────────────
# Main download orchestrator
# ─────────────────────────────────────────────────────────────────────


def download_robots(
    names: Optional[List[str]] = None,
    category: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download robot model assets from their respective sources.

    Download strategy (in order of preference):

    1. **robot_descriptions** (PyPI) — auto-discovers which modules map to
       which Menagerie directories at runtime. No hardcoded list.
    2. **git clone fallback** — shallow-clones Menagerie if
       ``robot_descriptions`` is not installed or has no matching module.
    3. **Custom GitHub repos** — for robots with ``asset.source`` in the
       registry (e.g. asimov_v0, reachy_mini, open_duck_mini).

    Args:
        names: List of robot names to download (None = all sim robots).
        category: Filter by category (arm, humanoid, mobile, ...).
        force: Re-download even if already present.

    Returns:
        Dict with downloaded/skipped/failed counts and details.
    """
    dest_dir = get_user_assets_dir()

    # Build robot list
    all_sim = {
        r["name"]: get_robot(r["name"])
        for r in registry_list_robots(mode="sim")
    }

    if names:
        resolved = {}
        for n in names:
            canonical = resolve_robot_name(n)
            if canonical in all_sim:
                resolved[canonical] = all_sim[canonical]
            else:
                logger.warning(
                    "Unknown robot: %s (resolved: %s)", n, canonical
                )
        robots = resolved
    elif category:
        robots = {
            n: info
            for n, info in all_sim.items()
            if info and info.get("category") == category
        }
    else:
        robots = {n: info for n, info in all_sim.items() if info}

    if not robots:
        return {
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "message": "No matching robots found.",
        }

    # Partition into need-download vs skip
    to_download = {}
    skipped = []
    for name, info in robots.items():
        if _needs_download(name, info, force):
            to_download[name] = info
        else:
            skipped.append(name)

    if not to_download:
        return {
            "downloaded": 0,
            "skipped": len(skipped),
            "failed": 0,
            "skipped_names": skipped,
            "message": (
                f"All {len(robots)} robots already have assets. "
                "Use force=True to re-download."
            ),
        }

    # Partition by source type
    menagerie_robots: Dict[str, dict] = {}
    github_robots: Dict[str, dict] = {}
    for name, info in to_download.items():
        source = _get_source(info)
        if source["type"] == "menagerie":
            menagerie_robots[name] = info
        elif source["type"] == "github":
            github_robots[name] = info
        else:
            logger.warning(
                "Unknown source type for %s: %s", name, source["type"]
            )

    # Download Menagerie robots
    results: Dict[str, str] = {}
    if menagerie_robots:
        use_rd = _robot_descriptions_available()
        if use_rd:
            rd_mapping = _discover_rd_mapping()
            logger.info(
                "Downloading %d robots via robot_descriptions "
                "(%d modules discovered)...",
                len(menagerie_robots),
                len(rd_mapping),
            )
            results.update(
                _download_via_robot_descriptions(menagerie_robots, dest_dir)
            )
            # Any that failed or were skipped → retry with git
            retry = {
                n: menagerie_robots[n]
                for n, r in results.items()
                if r.startswith("failed") or r.startswith("skipped")
            }
            if retry:
                logger.info(
                    "Retrying %d robots via git clone fallback...",
                    len(retry),
                )
                git_results = _download_from_menagerie_git(
                    retry, dest_dir
                )
                results.update(git_results)
        else:
            logger.info(
                "robot_descriptions not installed — using git clone for "
                "%d robots. Install with: pip install robot_descriptions",
                len(menagerie_robots),
            )
            results.update(
                _download_from_menagerie_git(menagerie_robots, dest_dir)
            )

    # Download custom GitHub robots
    for name, info in github_robots.items():
        result = _download_from_github(name, info, dest_dir)
        results[name] = result

    # Summarize
    downloaded = [n for n, r in results.items() if r == "downloaded"]
    failed = [(n, r) for n, r in results.items() if r != "downloaded"]

    method = (
        "robot_descriptions"
        if _robot_descriptions_available()
        else "git clone"
    )

    return {
        "downloaded": len(downloaded),
        "skipped": len(skipped),
        "failed": len(failed),
        "downloaded_names": downloaded,
        "skipped_names": skipped,
        "failed_names": [n for n, _ in failed],
        "failed_details": {n: r for n, r in failed},
        "assets_dir": str(dest_dir),
        "method": method,
        "message": (
            f"{len(downloaded)} downloaded ({method}), "
            f"{len(skipped)} already present, {len(failed)} failed."
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Strands Agent Tool
# ─────────────────────────────────────────────────────────────────────


@tool
def download_assets(
    action: str = "download",
    robots: str = None,
    category: str = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download and manage robot model assets (MJCF XML + meshes).

    Uses the ``robot_descriptions`` PyPI package (recommended by MuJoCo
    Menagerie) to download assets.  Falls back to git clone if not installed.
    Assets are cached in ~/.strands_robots/assets/.

    32 robots: arms (SO-100, Panda, UR5e), bimanual (ALOHA), hands (Shadow,
    LEAP), humanoids (G1, H1, Apollo, Cassie, Asimov), mobile (Spot, Go2).

    Args:
        action: Action to perform:
            - "download": Download robot assets (XML + meshes)
            - "list": List all available robots with status
            - "status": Show download status
        robots: Comma-separated robot names (e.g. "so100,panda,unitree_g1").
                Supports aliases. If omitted, downloads all.
        category: Filter: arm, bimanual, hand, humanoid, mobile, mobile_manip
        force: Force re-download even if present

    Returns:
        Dict with status and content
    """
    try:
        if action == "list":
            table = format_robot_table()
            return {
                "status": "success",
                "content": [
                    {"text": f"🤖 Available Robot Models:\n\n{table}"}
                ],
            }

        elif action == "status":
            from strands_robots.assets import list_available_robots

            robots_info = list_available_robots()
            available = sum(1 for r in robots_info if r["available"])
            missing = sum(1 for r in robots_info if not r["available"])

            rd_ok = _robot_descriptions_available()
            if rd_ok:
                rd_count = len(_discover_rd_mapping())
                method_str = (
                    f"✅ robot_descriptions installed "
                    f"({rd_count} modules discovered)"
                )
            else:
                method_str = (
                    "⚠️  robot_descriptions not installed "
                    "(using git clone fallback)"
                )

            lines = [
                f"📊 Asset Status: {available} available, "
                f"{missing} missing",
                f"📦 Download method: {method_str}\n",
            ]
            for r in robots_info:
                icon = "✅" if r["available"] else "❌"
                lines.append(
                    f"  {icon} {r['name']:<20s} {r['category']:<12s} "
                    f"{r['description']}"
                )
            if missing > 0:
                lines.append(
                    "\n💡 Download missing: "
                    "download_assets(action='download')"
                )
            if not rd_ok:
                lines.append(
                    "💡 For faster downloads: "
                    "pip install robot_descriptions"
                )
            lines.append(f"\n📁 User cache: {get_user_assets_dir()}")
            return {
                "status": "success",
                "content": [{"text": "\n".join(lines)}],
            }

        elif action == "download":
            robot_names = None
            if robots:
                robot_names = [
                    r.strip() for r in robots.split(",") if r.strip()
                ]
            result = download_robots(
                names=robot_names, category=category, force=force
            )

            text = "📦 Download Complete\n\n"
            text += f"✅ Downloaded: {result['downloaded']}\n"
            text += f"⏭️  Skipped: {result['skipped']}\n"
            text += f"❌ Failed: {result['failed']}\n"
            text += f"📦 Method: {result.get('method', '?')}\n"
            if result.get("downloaded_names"):
                text += (
                    "\nNewly downloaded: "
                    f"{', '.join(result['downloaded_names'])}"
                )
            if result.get("failed_details"):
                text += "\nFailures:"
                for n, reason in result["failed_details"].items():
                    text += f"\n  {n}: {reason}"
            text += f"\n\n📁 Assets: {result.get('assets_dir', '?')}"
            return {"status": "success", "content": [{"text": text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: {action}. "
                            "Valid: download, list, status"
                        )
                    }
                ],
            }

    except Exception as e:
        logger.error("download_assets error: %s", e)
        return {
            "status": "error",
            "content": [{"text": f"❌ Error: {str(e)}"}],
        }


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download robot assets "
            "(via robot_descriptions or git clone)"
        )
    )
    parser.add_argument(
        "robots", nargs="*", help="Robot names (default: all)"
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=[
            "arm",
            "bimanual",
            "hand",
            "humanoid",
            "mobile",
            "mobile_manip",
            "expressive",
        ],
    )
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument("--status", "-s", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(format_robot_table())
        return
    if args.status:
        result = download_assets(action="status")
        for c in result.get("content", []):
            print(c.get("text", ""))
        return

    result = download_robots(
        names=args.robots if args.robots else None,
        category=args.category,
        force=args.force,
    )
    print(result["message"])
    if result.get("failed_details"):
        for n, reason in result["failed_details"].items():
            print(f"  ❌ {n}: {reason}")


if __name__ == "__main__":
    main()
