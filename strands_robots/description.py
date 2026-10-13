"""What a robot *is*, independent of any world: its model file, joints, actuators, limits, keyframes.

The registry (``registry/robots.json`` + ``registry/urdf_robots.json``) maps a
name or alias to an asset shipped by ``robot_descriptions`` (MuJoCo Menagerie and
friends). ``Description.load("so101")`` resolves the file; the joint/actuator
facts are read from the model itself — via MuJoCo when it is installed, via a
plain XML walk when it is not — so a hardware-only install still knows its
``action_keys``.
"""

from __future__ import annotations

import importlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_REGISTRY_DIR = Path(__file__).parent / "registry"
_NAME_RE = re.compile(r"^[a-z0-9_.]+$")
_SCENE_CANDIDATES = ("scene.xml", "scene_mjx.xml")


def normalize_name(name: str) -> str:
    """lowercase, trimmed, dashes read as underscores — the one rule every lookup uses."""
    if not isinstance(name, str):
        raise TypeError(f"robot name must be a str, got {type(name).__name__}")
    return name.strip().lower().replace("-", "_")


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    """The curated registry: canonical name -> entry (aliases folded in as extra keys)."""
    robots: dict[str, dict[str, Any]] = json.loads((_REGISTRY_DIR / "robots.json").read_text())["robots"]
    out: dict[str, dict[str, Any]] = {}
    for key, entry in robots.items():
        entry = {**entry, "name": key}
        out[key] = entry
        for alias in entry.get("aliases", ()):
            out.setdefault(normalize_name(alias), entry)
    return out


@lru_cache(maxsize=1)
def urdf_registry() -> dict[str, dict[str, Any]]:
    return json.loads((_REGISTRY_DIR / "urdf_robots.json").read_text())["robots"]


def _import_description(module: str) -> Any:
    """Import a ``robot_descriptions`` module (clones the asset on first use)."""
    try:
        return importlib.import_module(f"robot_descriptions.{module}")
    except ModuleNotFoundError as exc:
        if exc.name == "robot_descriptions":
            raise ImportError("robot_descriptions is not installed: pip install robot_descriptions") from exc
        raise


def _discovered_module(norm: str) -> tuple[str, str] | None:
    """(module, format) from robot_descriptions' static table, MJCF preferred — no import, no network."""
    try:
        from robot_descriptions._descriptions import DESCRIPTIONS, Format
    except ImportError:
        return None
    for suffix, fmt in (("_mj_description", "mjcf"), ("_description", "urdf")):
        mod = f"{norm}{suffix}"
        desc = DESCRIPTIONS.get(mod)
        if desc is not None and (Format.MJCF if fmt == "mjcf" else Format.URDF) in desc.formats:
            return mod, fmt
    return None


def _resolve_asset(norm: str, entry: dict[str, Any] | None) -> tuple[Path, str, dict[str, Any]]:
    """Return (model_path, format, entry) for a registry entry, a discoverable name or a file path."""
    if entry is not None and "asset" in entry:
        module = entry["asset"].get("robot_descriptions_module")
        if module:
            mod = _import_description(module)
            path = getattr(mod, "MJCF_PATH", None) or getattr(mod, "URDF_PATH", None)
            if path is None:
                raise FileNotFoundError(f"robot_descriptions.{module} has neither MJCF_PATH nor URDF_PATH")
            return Path(path), ("mjcf" if module.endswith("_mj_description") else "urdf"), entry
        if (entry["asset"].get("source") or {}).get("type") == "github":
            from . import assets

            path = assets.resolve(entry)
            return path, ("urdf" if path.suffix == ".urdf" else "mjcf"), entry
    if norm in urdf_registry():
        u = urdf_registry()[norm]
        mod = _import_description(u["module"])
        return Path(mod.URDF_PATH), "urdf", {"name": norm, "category": u.get("category", "urdf"), **(entry or {})}
    found = _discovered_module(norm)
    if found is not None:
        module, fmt = found
        mod = _import_description(module)
        path = mod.MJCF_PATH if fmt == "mjcf" else mod.URDF_PATH
        return Path(path), fmt, {"name": norm, "category": "discovered", "discovered": True}
    if entry is not None:
        hw = (entry.get("hardware") or {}).get("lerobot_type")
        raise FileNotFoundError(
            f"robot {norm!r} is in the registry but ships no fetchable simulation model"
            + (f" (hardware only: lerobot_type {hw!r})" if hw else "")
        )
    raise KeyError(
        f"unknown robot {norm!r}: not in the registry, not a robot_descriptions module "
        f"({norm}_mj_description / {norm}_description) and not a path to an MJCF/URDF file"
    )


@dataclass(frozen=True)
class Description:
    """A robot model resolved to a file plus the facts a world or a bus needs about it."""

    name: str
    model_path: Path
    format: str  # "mjcf" | "urdf"
    category: str = "unknown"
    joints: tuple[str, ...] = ()  # hinge/slide joints, model order, labelled
    actuators: tuple[str, ...] = ()  # actuator names, model order
    action_keys: tuple[str, ...] = ()  # the joint each actuator drives, labelled, actuator order
    limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # joint -> (lo, hi) in rad
    keyframes: tuple[str, ...] = ()
    floating: bool = False
    joint_labels: Mapping[str, str] = field(default_factory=dict)  # model joint name -> label
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hardware: Mapping[str, Any] = field(default_factory=dict)
    gripper: Mapping[str, Any] = field(default_factory=dict)
    scene_path: Path | None = None

    @property
    def package_dir(self) -> Path:
        return self.model_path.parent

    @property
    def dof(self) -> int:
        return len(self.action_keys)

    def label(self, model_joint: str) -> str:
        return self.joint_labels.get(model_joint, model_joint)

    def unlabel(self, label: str) -> str:
        for k, v in self.joint_labels.items():
            if v == label:
                return k
        return label

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary (what an agent tool and the dashboard read)."""
        return {
            "name": self.name,
            "category": self.category,
            "format": self.format,
            "model": str(self.model_path),
            "dof": self.dof,
            "action_keys": list(self.action_keys),
            "limits": {k: [float(a), float(b)] for k, (a, b) in self.limits.items()},
            "keyframes": list(self.keyframes),
            "floating": self.floating,
        }

    # loading

    @classmethod
    def load(cls, name: str | Path) -> Description:
        """Resolve a registry name, alias, ``robot_descriptions`` name or MJCF/URDF path."""
        if isinstance(name, Path) or name.endswith((".xml", ".urdf", ".mjcf")):
            path = Path(name).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"robot model file not found: {path}")
            fmt = "urdf" if path.suffix == ".urdf" else "mjcf"
            entry: dict[str, Any] = {"name": path.stem, "category": "file"}
        else:
            norm = normalize_name(name)
            if not _NAME_RE.match(norm):
                raise ValueError(f"robot name {name!r} may only contain letters, digits, '_' and '.'")
            entry = registry().get(norm)  # type: ignore[assignment]
            path, fmt, entry = _resolve_asset(norm, entry)
        return cls._from_file(path, fmt, entry)

    @classmethod
    def _from_file(cls, path: Path, fmt: str, entry: dict[str, Any]) -> Description:
        labels_raw = entry.get("joint_labels") or {}
        scene = None
        if fmt == "mjcf":
            asset = entry.get("asset") or {}
            root = path.parents[len(Path(asset["model_xml"]).parts) - 1] if "model_xml" in asset else path.parent
            for cand in (path.parent / c for c in _SCENE_CANDIDATES):
                if cand.is_file():
                    scene = cand
                    break
            if scene is None and asset.get("scene_xml") and (root / asset["scene_xml"]).is_file():
                scene = root / asset["scene_xml"]
        facts = _introspect(path, fmt, scene)
        labels = {j: labels_raw[j] for j in facts["joints"] if j in labels_raw}

        def lab(j: str) -> str:
            return labels.get(j, j)

        sp = entry.get("spawn_position") or (0.0, 0.0, 0.0)
        return cls(
            name=entry.get("name", path.stem),
            model_path=path,
            format=fmt,
            category=entry.get("category", "unknown"),
            joints=tuple(lab(j) for j in facts["joints"]),
            actuators=tuple(facts["actuators"]),
            action_keys=tuple(lab(j) for j in facts["action_joints"]),
            limits={lab(j): lim for j, lim in facts["limits"].items()},
            keyframes=tuple(facts["keyframes"]),
            floating=facts["floating"],
            joint_labels=labels,
            spawn_position=(float(sp[0]), float(sp[1]), float(sp[2])),
            hardware=dict(entry.get("hardware") or {}),
            gripper=dict(entry.get("gripper") or {}),
            scene_path=scene,
        )


def _introspect(path: Path, fmt: str, scene: Path | None = None) -> dict[str, Any]:
    """Joints, actuators, limits, keyframes of a model file: MuJoCo if available, else an XML walk.

    A robot file that only compiles inside its scene (LeKiwi's wheel contacts name the scene's floor)
    is read through the scene.
    """
    from ._gl import ensure_gl

    ensure_gl()
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return _introspect_xml(path, fmt)
    try:
        return _introspect_mujoco(path)
    except ValueError:
        if scene is None:
            raise
        return _introspect_mujoco(scene)


def _introspect_mujoco(path: Path) -> dict[str, Any]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(path))
    joints, limits, floating = [], {}, False
    for i in range(m.njnt):
        jt = int(m.jnt_type[i])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            floating = True
            continue
        if jt == mujoco.mjtJoint.mjJNT_BALL:
            continue
        name = m.joint(i).name or f"joint{i}"
        joints.append(name)
        if m.jnt_limited[i]:
            limits[name] = (float(m.jnt_range[i, 0]), float(m.jnt_range[i, 1]))
    actuators, action_joints = [], []
    for i in range(m.nu):
        actuators.append(m.actuator(i).name or f"actuator{i}")
        if int(m.actuator_trntype[i]) == mujoco.mjtTrn.mjTRN_JOINT:
            action_joints.append(m.joint(int(m.actuator_trnid[i, 0])).name)
        else:
            action_joints.append(actuators[-1])
    return {
        "joints": joints,
        "actuators": actuators,
        "action_joints": action_joints,
        "limits": limits,
        "keyframes": [m.key(i).name or f"key{i}" for i in range(m.nkey)],
        "floating": floating,
    }


def _introspect_xml(path: Path, fmt: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    joints: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    floating = False
    if fmt == "urdf":
        for j in root.iter("joint"):
            jt = j.get("type", "revolute")
            if jt == "fixed":
                continue
            if jt == "floating":
                floating = True
                continue
            name = j.get("name", f"joint{len(joints)}")
            joints.append(name)
            lim = j.find("limit")
            if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
                limits[name] = (float(lim.get("lower", 0)), float(lim.get("upper", 0)))
        return {
            "joints": joints,
            "actuators": list(joints),
            "action_joints": list(joints),
            "limits": limits,
            "keyframes": [],
            "floating": floating,
        }
    for inc in root.iter("include"):  # one level of <include file=...>
        sub = path.parent / inc.get("file", "")
        if sub.is_file():
            root.extend(list(ET.parse(sub).getroot()))
    for j in root.iter("joint"):
        jt = j.get("type", "hinge")
        if jt == "free":
            floating = True
            continue
        if jt == "ball":
            continue
        name = j.get("name", f"joint{len(joints)}")
        joints.append(name)
        if j.get("range"):
            lo, hi = (float(x) for x in j.get("range", "0 0").split()[:2])
            limits[name] = (lo, hi)
    if root.find(".//freejoint") is not None:
        floating = True
    actuators, action_joints = [], []
    act = root.find("actuator")
    for i, a in enumerate(list(act) if act is not None else []):
        actuators.append(a.get("name", f"actuator{i}"))
        action_joints.append(a.get("joint", actuators[-1]))
    kf = root.find("keyframe")
    keyframes = [k.get("name", f"key{i}") for i, k in enumerate(list(kf) if kf is not None else [])]
    return {
        "joints": joints,
        "actuators": actuators,
        "action_joints": action_joints,
        "limits": limits,
        "keyframes": keyframes,
        "floating": floating,
    }
