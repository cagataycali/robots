"""mkdocs hook: the viewer manifest, generated from the robot registry.

``docs/assets/viewer/robots.json`` is written at every build from
``strands_robots/registry/robots.json``. The browser viewer reads it to know,
for each robot, where its MJCF and meshes stream from (jsDelivr in front of the
model's public git repo, pinned to a commit), which scene file to load, and what
the catalog card should say. Nothing about a robot is typed twice.

Filesystem only: no ``strands_robots`` import.
"""

from __future__ import annotations

import json
import logging
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
_GITHUB_REFS: dict[str, str] = {}


def _lfs_url(name: str, asset: dict) -> str | None:
    """GitHub's LFS media endpoint for the same directory; jsDelivr serves LFS pointers, not blobs."""
    source = asset.get("source") or {"type": "menagerie"}
    if source.get("type") == "menagerie":
        return f"https://media.githubusercontent.com/media/google-deepmind/mujoco_menagerie/{MENAGERIE_REF}/{asset['dir']}/"
    if source.get("type") == "github" and source.get("repo"):
        ref = _GITHUB_REFS.get(name, source.get("ref") or "main")
        subdir = source.get("subdir", "").strip("/")
        return f"https://media.githubusercontent.com/media/{source['repo']}/{ref}/{subdir}/" if subdir else f"https://media.githubusercontent.com/media/{source['repo']}/{ref}/"
    return None


def _base_url(name: str, asset: dict) -> str | None:
    """Where the robot's asset directory is served from, with a trailing slash."""
    source = asset.get("source") or {"type": "menagerie"}
    if source.get("type") == "menagerie":
        return f"{_CDN}/google-deepmind/mujoco_menagerie@{MENAGERIE_REF}/{asset['dir']}/"
    if source.get("type") == "github" and source.get("repo"):
        ref = _GITHUB_REFS.get(name, source.get("ref") or "main")
        subdir = source.get("subdir", "").strip("/")
        return f"{_CDN}/{source['repo']}@{ref}/{subdir}/" if subdir else f"{_CDN}/{source['repo']}@{ref}/"
    return None


def build_manifest() -> dict:
    """Derive the manifest; pure function of the registry and docs/assets."""
    robots = json.loads(_REGISTRY.read_text(encoding="utf-8"))["robots"]
    out: dict[str, dict] = {}
    for name, spec in robots.items():
        asset = spec.get("asset") or {}
        base = _base_url(name, asset) if asset else None
        thumb = _DOCS / "assets" / "img" / "robots" / f"{name}.webp"
        entry = {
            "name": name,
            "description": spec.get("description", name),
            "category": spec["category"],
            "joints": spec.get("joints"),
            "aliases": list(spec.get("aliases", ())),
            "sim": bool(base),
            "real": bool(spec.get("hardware")),
            "driver": (spec.get("hardware") or {}).get("driver"),
            "base_url": base,
            "lfs_url": _lfs_url(name, asset) if base else None,
            "scene": asset.get("scene_xml") if base else None,
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
    log.info("viewer manifest: %d robots, %d renderable", len(manifest["robots"]), sum(1 for r in manifest["robots"].values() if r["sim"]))


if __name__ == "__main__":
    m = build_manifest()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(m['robots'])} robots, {sum(1 for r in m['robots'].values() if r['sim'])} renderable -> {_OUT}")
