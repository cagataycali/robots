"""mkdocs hook: the viewer manifest, generated from the robot registry.

``docs/assets/viewer/robots.json`` is written at every build from
``strands_robots/registry/robots.json`` and the URDF tail in ``urdf_robots.json``
(merged by ``registry_view.py``). The browser viewer reads it to know,
for each robot, where its MJCF and meshes stream from (jsDelivr in front of the
model's public git repo, pinned to a commit), which scene file to load, and what
the catalog card should say. Nothing about a robot is typed twice.

Filesystem only: no ``strands_robots`` import.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.manifest")

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "strands_robots" / "registry" / "robots.json"
_DOCS = _REPO / "docs"
_OUT = _DOCS / "assets" / "viewer" / "robots.json"

#: google-deepmind/mujoco_menagerie main on 2026-09-23. Bump deliberately.
MENAGERIE_REF = "c96a32d28fb5da84da38c1da4d749e7a13212855"
_CDN = "https://cdn.jsdelivr.net/gh"

#: Refs for the robots whose models live outside the menagerie. ``main`` is a
#: moving target on jsDelivr (cached up to 12 h); pin when the upstream tags.
_GITHUB_REFS: dict[str, str] = {
    # default-branch heads read 2026-09-27; jsDelivr cannot resolve a branch literally
    # named "v2" (Open_Duck_Mini), so every github-source robot is pinned by commit.
    "microduck": "cb70b792312d559a4da09064d92009079671815f",
    "reachy_mini": "292b2434cadbb3ff932863bd9b476741bb6ef2fd",
    "lekiwi": "32cf6a69eb320cc22620cdaa529e35f20fc12b1f",
    "yahboom_m3pro": "bdac682e57a8eeaf7b18eece44bfbfc54de98e8a",
    "open_duck_mini": "b23317a485b3cec7d8417f352478778b3475173c",
    "asimov_v0": "759204f531b071e65540e8b31d89736a3d09e0dd",
}

#: Repositories that moved since the registry was written (GitHub redirects the API,
#: jsDelivr does not).
_REPO_MOVES: dict[str, str] = {"asimovinc/asimov-v0": "menloresearch/asimov-v0"}

#: Registry scene files that do not exist upstream; the viewer loads the file that does.
#: aero_hand: the registry says scene_left.xml, the menagerie ships scene_right.xml.
_SCENE_OVERRIDES: dict[str, str] = {"aero_hand": "scene_right.xml"}

#: Models the browser build cannot compile, with the reason the viewer shows.
_VIEWER_UNSUPPORTED: dict[str, str] = {
    "rby1": "its MJCF includes the same file twice with merge=true, which the WebAssembly build refuses",
}

#: Shown for a robot_descriptions URDF robot: its MJCF is compiled by the loader
#: on the user's machine, so there is nothing pinned upstream for the browser to load.
_URDF_VIEWER_NOTE = (
    "compiled from its robot_descriptions URDF on first use; the meshes are converted "
    "locally, so there is no upstream MJCF to stream and the thumbnail is a local render"
)

#: Robots whose registry entry names a ``robot_descriptions`` module instead of a
#: menagerie directory. robot_descriptions pins each repository to a commit; these
#: are those pins (robot_descriptions 1.x, read 2026-09-27) with the path of the
#: asset directory inside the repository. The hook cannot import robot_descriptions
#: (filesystem only), so the table is explicit and reviewable.
_DESCRIPTION_REPOS: dict[str, tuple[str, str, str]] = {
    # name: (owner/repo, commit, subdir)
    "openarm": ("enactic/openarm_mujoco", "cd30dd4c0a97832d1c063bf759514ed18fbe04a5", "v1"),
    "ability_hand": (
        "psyonicinc/ability-hand-api",
        "89407424edfc22faceaedcd7c3ea2b7947cbbb2c",
        "python/ah_simulators/mujoco_xml",
    ),
    "elf2": ("bxirobotics/robot_models", "eabe24ce937f8e633077a163b883e92e8996c36e", "elf2_dof25/xml"),
    "jvrc": ("isri-aist/jvrc_mj_description", "0f0ce7daefdd66c54e0909a6bf2c22154844f5f3", ""),
    "rby1": ("uynitsuj/rby1_description", "e4c07203aa0a0d1b6b3b39da105cb00a77e2bc72", "models/rby1a/mujoco"),
    "unitree_h1_2": (
        "unitreerobotics/unitree_ros",
        "267182b8521c8d6a631bab1fe63836873237a525",
        "robots/h1_2_description",
    ),
    "aliengo": ("unitreerobotics/unitree_mujoco", "f3300ff1bf0ab9efbea0162717353480d9b05d73", "data/aliengo"),
    "unitree_a1": ("unitreerobotics/unitree_mujoco", "f3300ff1bf0ab9efbea0162717353480d9b05d73", "data/a1"),
}


def _lfs_url(name: str, asset: dict) -> str | None:
    """GitHub's LFS media endpoint for the same directory; jsDelivr serves LFS pointers, not blobs."""
    if name in _DESCRIPTION_REPOS:
        repo, commit, subdir = _DESCRIPTION_REPOS[name]
        return (
            f"https://media.githubusercontent.com/media/{repo}/{commit}/{subdir}/"
            if subdir
            else f"https://media.githubusercontent.com/media/{repo}/{commit}/"
        )
    source = asset.get("source") or {"type": "menagerie"}
    if source.get("type") == "menagerie":
        return f"https://media.githubusercontent.com/media/google-deepmind/mujoco_menagerie/{MENAGERIE_REF}/{asset['dir']}/"
    if source.get("type") == "github" and source.get("repo"):
        repo = _REPO_MOVES.get(source["repo"], source["repo"])
        ref = _GITHUB_REFS.get(name, source.get("ref") or "main")
        subdir = source.get("subdir", "").strip("/")
        return (
            f"https://media.githubusercontent.com/media/{repo}/{ref}/{subdir}/"
            if subdir
            else f"https://media.githubusercontent.com/media/{repo}/{ref}/"
        )
    return None


def _base_url(name: str, asset: dict) -> str | None:
    """Where the robot's asset directory is served from, with a trailing slash."""
    if name in _DESCRIPTION_REPOS:
        repo, commit, subdir = _DESCRIPTION_REPOS[name]
        return f"{_CDN}/{repo}@{commit}/{subdir}/" if subdir else f"{_CDN}/{repo}@{commit}/"
    source = asset.get("source") or {"type": "menagerie"}
    if source.get("type") == "menagerie":
        return f"{_CDN}/google-deepmind/mujoco_menagerie@{MENAGERIE_REF}/{asset['dir']}/"
    if source.get("type") == "github" and source.get("repo"):
        repo = _REPO_MOVES.get(source["repo"], source["repo"])
        ref = _GITHUB_REFS.get(name, source.get("ref") or "main")
        subdir = source.get("subdir", "").strip("/")
        return f"{_CDN}/{repo}@{ref}/{subdir}/" if subdir else f"{_CDN}/{repo}@{ref}/"
    return None


def _raw_url(base: str | None) -> str | None:
    """raw.githubusercontent.com mirror of a jsDelivr gh base (no 20 MB per-file limit)."""
    if not base:
        return None
    rest = base[len(_CDN) + 1 :]  # owner/repo@ref/subdir/
    repo, _, tail = rest.partition("@")
    ref, _, subdir = tail.partition("/")
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{subdir}"


def build_manifest() -> dict:
    """Derive the manifest; pure function of the registry and docs/assets."""
    robots = _registry_view().merged()
    out: dict[str, dict] = {}
    for name, spec in robots.items():
        asset = spec.get("asset") or {}
        urdf = spec.get("source") == "urdf"
        # A URDF robot's MJCF exists only where the loader wrote it: nothing
        # upstream serves it, so the browser gets the local thumbnail.
        base = _base_url(name, asset) if asset and not urdf else None
        thumb = _DOCS / "assets" / "img" / "robots" / f"{name}.webp"
        entry = {
            "name": name,
            "description": spec.get("description", name),
            "category": spec["category"],
            "joints": spec.get("joints"),
            "aliases": list(spec.get("aliases", ())),
            "sim": bool(asset),
            "viewer": bool(base) and name not in _VIEWER_UNSUPPORTED,
            "viewer_note": _URDF_VIEWER_NOTE if urdf and asset else _VIEWER_UNSUPPORTED.get(name),
            "source": spec.get("source", "curated"),
            "real": bool(spec.get("hardware")),
            "driver": (spec.get("hardware") or {}).get("driver"),
            "base_url": base,
            "lfs_url": _lfs_url(name, asset) if base else None,
            "raw_url": _raw_url(base),
            "scene": _SCENE_OVERRIDES.get(name, asset.get("scene_xml")) if base else None,
            "model": asset.get("model_xml") if base else None,
            "thumbnail": f"assets/img/robots/{name}.webp" if thumb.exists() else None,
        }
        out[name] = entry
    return {"menagerie_ref": MENAGERIE_REF, "robots": out}


def on_pre_build(config) -> None:  # noqa: ANN001 - mkdocs signature
    """Write the manifest before the files are collected so it ships as an asset."""
    manifest = build_manifest()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "viewer manifest: %d robots, %d renderable",
        len(manifest["robots"]),
        sum(1 for r in manifest["robots"].values() if r["viewer"]),
    )


if __name__ == "__main__":
    m = build_manifest()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(m['robots'])} robots, {sum(1 for r in m['robots'].values() if r['viewer'])} renderable -> {_OUT}")


def _registry_view():  # noqa: ANN202 - a sibling hook module, loaded by path like the others
    """``docs/hooks/registry_view.py``: robots.json merged with the URDF long tail."""
    name = "docs_hooks_registry_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "registry_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
