# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deploy contract of an Isaac Lab policy: what its inputs and outputs mean.

An exported actor is a function from one float vector to another. What the
numbers mean lives in the Isaac Lab environment it was trained in, not in the
weights: which joint each output drives (in the ORDER the run's articulation
reported, which differs between the PhysX and Newton presets of the same
task), the ``scale`` and default-pose ``offset`` the action term applies
(``target = offset + scale * action``), the control period, and which
observation terms, in which frames and units, the input vector is
concatenated from. Deploying without them binds outputs by position and
commands raw network values as joint angles: the Go2 rough-terrain policy,
correct to 0.0 against Isaac Lab's own TorchScript export, commanded joints up
to 1.73 rad from where Isaac Lab would have put them, and the robot fell.

Isaac Lab writes all of it when a run is launched with
``--export_io_descriptors`` (``<run>/io_descriptors/IO_descriptors.yaml``).
:func:`contract_from_io_descriptors` turns that file into the JSON-safe dict
``policy_meta.json`` carries under ``deploy_contract``, and
:func:`apply_action_contract` is the one place a raw action becomes joint
targets. Nothing here imports Isaac Lab, torch or YAML.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Bumped when a field changes meaning; a reader refuses a version it does not know.
CONTRACT_VERSION = 1

#: Action terms whose processed action is a joint POSITION target, which is
#: what a strands ``send_action`` commands. Other terms (effort, velocity,
#: task-space, binary gripper) are recorded but refused at deploy.
POSITION_ACTION_TERMS: frozenset[str] = frozenset(
    {
        "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
    }
)

#: Isaac Lab 3.0 writes quaternions scalar-last and ``base_lin_vel`` /
#: ``base_ang_vel`` in the root (body) frame; strands reports ``base_quat`` as
#: ``[w, x, y, z]`` and base velocities in the world frame. The contract names
#: both so a caller building the observation cannot assume one convention.
ISAACLAB_QUAT_ORDER = "xyzw"
ISAACLAB_BASE_VELOCITY_FRAME = "body"


class DeployContractError(ValueError):
    """An Isaac Lab IO descriptor, or a policy's use of one, that cannot be honoured."""


def _floats(values: Any, *, count: int, what: str) -> list[float]:
    """*values* as *count* floats; a scalar is broadcast."""
    if isinstance(values, int | float) and not isinstance(values, bool):
        return [float(values)] * count
    if isinstance(values, Sequence) and not isinstance(values, str) and len(values) == count:
        return [float(v) for v in values]
    raise DeployContractError(f"{what}: expected a number or {count} numbers, got {values!r}")


def _width(term: Mapping[str, Any]) -> int:
    shape = term.get("shape") or []
    width = math.prod(int(d) for d in shape) if shape else 0
    overloads = term.get("overloads") or {}
    history = int(overloads.get("history_length") or 0)
    if history > 0 and overloads.get("flatten_history_dim", True):
        width *= history
    return width


def contract_from_io_descriptors(
    descriptors: Mapping[str, Any], *, physics: str | None = None, group: str = "policy"
) -> dict[str, Any]:
    """Build the deploy contract from a parsed ``IO_descriptors.yaml``.

    Args:
        descriptors: The YAML file, parsed (``actions``, ``observations``,
            ``articulations``, ``scene``).
        physics: The physics preset the run trained on, recorded because the
            joint order it produced is only valid for that preset.
        group: The observation group the actor reads (Isaac Lab's ``policy``).

    Returns:
        A JSON-safe dict: ``version``, ``action_keys`` (every term's joints,
        concatenated in the run's order), ``action_terms`` (per term: ``name``,
        ``type``, ``joint_names``, ``scale``, ``offset``, ``clip``, ``start``,
        ``width``), ``obs_layout`` (per term: ``name``, ``type``, ``start``,
        ``width``, ``units``, and ``joint_names`` / ``joint_offsets`` where the
        term has them), ``num_obs``, ``control_dt``, ``physics_dt``,
        ``decimation``, ``default_joint_pos`` (joint name -> rad), ``physics``,
        ``quat_order`` and ``base_velocity_frame``.

    Raises:
        DeployContractError: The file has no action terms or no *group*
            observations, or a term's scale/offset has the wrong length.
    """
    actions = descriptors.get("actions") or []
    if not actions:
        raise DeployContractError("the IO descriptors list no action terms")
    terms: list[dict[str, Any]] = []
    action_keys: list[str] = []
    for term in actions:
        joints = [str(j) for j in term.get("joint_names") or []]
        width = _width(term) or len(joints)
        if joints and len(joints) != width:
            raise DeployContractError(
                f"action term {term.get('name')!r} names {len(joints)} joints for {width} outputs"
            )
        offset = term.get("offset")
        clip = term.get("clip")
        terms.append(
            {
                "name": str(term.get("name", "")),
                "type": str(term.get("full_path", term.get("action_type", ""))),
                "joint_names": joints,
                "start": sum(t["width"] for t in terms),
                "width": width,
                "scale": _floats(
                    term.get("scale", 1.0) if term.get("scale") is not None else 1.0, count=width, what="scale"
                ),
                "offset": _floats(offset if offset is not None else 0.0, count=width, what="offset"),
                "clip": _clip_list(clip, joints, width),
            }
        )
        action_keys += joints
    observations = (descriptors.get("observations") or {}).get(group) or []
    if not observations:
        raise DeployContractError(f"the IO descriptors list no {group!r} observation terms")
    layout: list[dict[str, Any]] = []
    start = 0
    for term in observations:
        width = _width(term)
        entry: dict[str, Any] = {
            "name": str(term.get("name", "")),
            "type": str(term.get("full_path", "")),
            "start": start,
            "width": width,
            "units": (term.get("extras") or {}).get("units"),
        }
        if term.get("joint_names"):
            entry["joint_names"] = [str(j) for j in term["joint_names"]]
        if term.get("joint_pos_offsets") is not None:
            entry["joint_offsets"] = [float(v) for v in term["joint_pos_offsets"]]
        scale = (term.get("overloads") or {}).get("scale")
        if scale is not None:
            entry["scale"] = scale
        layout.append(entry)
        start += width
    scene = descriptors.get("scene") or {}
    robot = (descriptors.get("articulations") or {}).get("robot") or {}
    names = [str(j) for j in robot.get("joint_names") or []]
    default = robot.get("default_joint_pos") or []
    return {
        "version": CONTRACT_VERSION,
        "source": "isaaclab_io_descriptors",
        "physics": physics,
        "action_keys": action_keys,
        "action_terms": terms,
        "obs_group": group,
        "obs_layout": layout,
        "num_obs": start,
        "control_dt": float(scene["dt"]) if "dt" in scene else None,
        "physics_dt": float(scene["physics_dt"]) if "physics_dt" in scene else None,
        "decimation": int(scene["decimation"]) if "decimation" in scene else None,
        "default_joint_pos": {n: float(v) for n, v in zip(names, default, strict=False)},
        "quat_order": ISAACLAB_QUAT_ORDER,
        "base_velocity_frame": ISAACLAB_BASE_VELOCITY_FRAME,
    }


def _clip_list(clip: Any, joints: list[str], width: int) -> list[list[float]] | None:
    """A term's clip as ``[[low, high]] * width``, or ``None`` when it has none."""
    if clip is None:
        return None
    if isinstance(clip, Mapping):
        # Isaac Lab keys a per-joint clip by joint-name pattern; resolved names only.
        missing = [j for j in joints if j not in clip]
        if missing:
            raise DeployContractError(f"action clip names patterns, not the joints {missing}")
        return [[float(clip[j][0]), float(clip[j][1])] for j in joints]
    if isinstance(clip, Sequence) and len(clip) == 2 and all(isinstance(v, int | float) for v in clip):
        return [[float(clip[0]), float(clip[1])]] * width
    raise DeployContractError(f"unrecognised action clip {clip!r}")


def contract_problems(contract: Mapping[str, Any], *, num_actor_obs: int, num_actions: int) -> list[str]:
    """What makes *contract* disagree with the actor it was exported beside; empty when it agrees."""
    problems: list[str] = []
    if contract.get("version") != CONTRACT_VERSION:
        problems.append(f"deploy contract version {contract.get('version')!r} is not {CONTRACT_VERSION}")
        return problems
    if len(contract.get("action_keys") or []) != num_actions:
        problems.append(
            f"the contract names {len(contract.get('action_keys') or [])} action joints, the actor emits {num_actions}"
        )
    # Fewer described values than the actor reads is Isaac Lab's own gap, not a
    # contradiction: its IO descriptors skip terms that have no descriptor (a
    # rough-terrain task's 187-value ``height_scan`` ray-cast sensor), so the
    # layout is marked incomplete by :func:`complete_obs_layout` instead. More
    # described values than the actor reads cannot come from the same run.
    if int(contract.get("num_obs") or 0) > num_actor_obs:
        problems.append(
            f"the contract's observation terms add up to {contract.get('num_obs')}, the actor reads {num_actor_obs}"
        )
    unsupported = sorted(
        {t["type"] for t in contract.get("action_terms", []) if t["type"] not in POSITION_ACTION_TERMS}
    )
    if unsupported:
        problems.append(
            f"action term(s) {unsupported} are not joint-position targets; strands commands joint positions only"
        )
    return problems


def _basename(func: Any) -> str:
    """``isaaclab.envs.mdp.observations:joint_pos_rel`` -> ``joint_pos_rel``."""
    return str(func or "").replace(":", ".").rsplit(".", 1)[-1]


def _per_joint(value: Any, joint: str) -> float | None:
    """A scalar, or the value of the first ``{regex: value}`` entry matching *joint*."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping):
        for pattern, entry in value.items():
            if re.fullmatch(str(pattern), joint) and isinstance(entry, int | float):
                return float(entry)
    return None


def env_cfg_terms(env_cfg: Mapping[str, Any], *, group: str = "policy") -> list[dict[str, Any]]:
    """The observation terms of *group* in the run's ``params/env.yaml``, in concatenation order."""
    terms = []
    for name, term in ((env_cfg.get("observations") or {}).get(group) or {}).items():
        if not isinstance(term, Mapping) or "func" not in term:
            continue
        params = {k: v for k, v in (term.get("params") or {}).items() if isinstance(v, int | float | str | bool)}
        terms.append(
            {
                "term": str(name),
                "func": _basename(term["func"]),
                "clip": term.get("clip"),
                "scale": term.get("scale"),
                "params": params,
            }  # fmt: skip
        )
    return terms


def env_cfg_actuators(env_cfg: Mapping[str, Any], joints: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per joint: the actuator model the run's articulation used (class, stiffness, damping, effort limit).

    Isaac Lab's Go2 is a ``DCMotor`` PD (stiffness 25, damping 0.5, effort
    23.7 / 45.43 N m); strands' MuJoCo Go2 has torque motors. A policy trained
    against the first sends position targets the second reads as torques, so
    the deploying engine needs this to close the same PD loop.
    """
    groups = (((env_cfg.get("scene") or {}).get("robot") or {}).get("actuators")) or {}
    out: dict[str, dict[str, Any]] = {}
    for joint in joints:
        for group in groups.values():
            if not isinstance(group, Mapping):
                continue
            if not any(re.fullmatch(str(p), joint) for p in group.get("joint_names_expr") or []):
                continue
            effort = next(
                (e for key in ("effort_limit_sim", "effort_limit", "actuator_effort_limit", "saturation_effort")
                 if (e := _per_joint(group.get(key), joint)) is not None),
                None,
            )  # fmt: skip
            out[joint] = {
                "model": _basename(group.get("class_type")),
                "stiffness": _per_joint(group.get("stiffness"), joint),
                "damping": _per_joint(group.get("damping"), joint),
                "effort_limit": effort,
            }
            break
    return out


def attach_env_cfg(contract: Mapping[str, Any], env_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """*contract* plus what only the run's ``params/env.yaml`` records: term order, clips and actuators."""
    if not env_cfg:
        return dict(contract)
    group = str(contract.get("obs_group") or "policy")
    return {
        **contract,
        "env_obs_terms": env_cfg_terms(env_cfg, group=group),
        "actuators": env_cfg_actuators(env_cfg, list(contract.get("action_keys") or [])),
    }


def _merge_env_terms(contract: Mapping[str, Any], unaccounted: int) -> list[dict[str, Any]] | None:
    """The full layout: env.yaml's term order, the descriptors' widths, the one undescribed term's gap."""
    described = list(contract.get("obs_layout") or [])
    env_terms = list(contract.get("env_obs_terms") or [])
    if not env_terms:
        return None
    layout, start, missing = [], 0, 0
    for term in env_terms:
        match = next((d for d in described if d["name"] == term["func"]), None)
        if match is not None:
            described.remove(match)
            width = int(match["width"])
            entry = {**match, **term}
        else:
            missing += 1
            width = unaccounted
            entry = {"name": term["func"], "type": term["func"], **term}
        entry.update(start=start, width=width)
        layout.append(entry)
        start += width
    if described or missing > 1 or (missing == 1 and unaccounted == 0) or (missing == 0 and unaccounted):
        return None
    return layout


def complete_obs_layout(contract: Mapping[str, Any], *, num_actor_obs: int) -> dict[str, Any]:
    """*contract* with ``obs_layout_complete`` and ``obs_unaccounted`` set against the actor's width.

    Isaac Lab describes only observation terms that carry a descriptor; a
    sensor term such as ``height_scan`` is absent from the file although the
    actor reads it. The action half of the contract is unaffected, so the
    export keeps it and says how many input values the layout leaves unnamed.
    """
    unaccounted = max(0, num_actor_obs - int(contract.get("num_obs") or 0))
    merged = _merge_env_terms(contract, unaccounted)
    if merged is not None and unaccounted:
        # The run's env.yaml names the term the descriptors skip, in order.
        return {**contract, "obs_layout": merged, "num_obs": num_actor_obs, "obs_layout_complete": True,
                "obs_unaccounted": 0}  # fmt: skip
    if merged is not None:
        contract = {**contract, "obs_layout": merged}
    return {**contract, "obs_layout_complete": unaccounted == 0, "obs_unaccounted": unaccounted}


def _rotate_to_body(quat_wxyz: Sequence[float], vec: Sequence[float]) -> list[float]:
    """``R(q)^T v``: a world-frame vector in the body frame of unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = (float(c) for c in quat_wxyz)
    vx, vy, vz = (float(c) for c in vec)
    # Rotation by the conjugate quaternion (w, -x, -y, -z).
    tx, ty, tz = 2 * (-y * vz + z * vy), 2 * (-z * vx + x * vz), 2 * (-x * vy + y * vx)
    return [vx + w * tx + (-y * tz + z * ty), vy + w * ty + (-z * tx + x * tz), vz + w * tz + (-x * ty + y * tx)]


def _vector(observation: Mapping[str, Any], key: str, width: int) -> list[float]:
    value = observation.get(key)
    if value is None or isinstance(value, str | bytes):
        raise DeployContractError(f"the observation has no {key!r}, which the actor's observation is built from")
    values = [float(v) for v in value]
    if len(values) != width:
        raise DeployContractError(f"observation {key!r} has {len(values)} values, expected {width}")
    return values


def build_policy_obs(
    contract: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    joint_keys: Mapping[str, str],
    last_action: Sequence[float],
    command: Sequence[float] | None = None,
    obs_terms: Mapping[str, Sequence[float]] | None = None,
) -> list[float]:
    """Assemble the actor's input vector from a strands observation, term by term, as Isaac Lab would.

    strands reports ``base_quat`` as ``[w, x, y, z]``, ``base_lin_vel`` in the
    WORLD frame and ``base_ang_vel`` in the BODY frame (MuJoCo's free-joint
    convention); Isaac Lab 3.0's terms want body-frame velocities and gravity
    projected into the body. Each supported term is computed in that frame:

    * ``base_lin_vel`` = R(q)^T v_world; ``base_ang_vel`` as reported;
      ``projected_gravity`` = R(q)^T (0, 0, -1);
    * ``generated_commands``: *command* (default: zeros = stand still);
    * ``joint_pos_rel`` = q - the run's default pose, ``joint_vel_rel`` = qd,
      read by joint name through *joint_keys*;
    * ``last_action``: *last_action*, the actor's previous raw output;
    * ``height_scan``: flat ground at z = 0 under the base, i.e.
      ``base_z - offset`` for every ray (Isaac Lab's default offset 0.5);
    * any other term: *obs_terms*[name] (by env term name or function name).

    The term's ``scale`` and ``clip`` from the run's config are then applied.

    Raises:
        DeployContractError: The layout is incomplete, a term is unknown and
            not supplied, or the observation lacks what a term reads.
    """
    if not contract.get("obs_layout_complete"):
        raise DeployContractError(
            f"the contract describes {contract.get('num_obs')} of the actor's input values "
            f"({contract.get('obs_unaccounted')} unaccounted), so the observation cannot be built; "
            "pass the whole vector as 'policy_obs'"
        )
    supplied = dict(obs_terms or {})
    out: list[float] = []
    for term in contract["obs_layout"]:
        func, name, width = (
            str(term.get("func") or term["name"]),
            str(term.get("term") or term["name"]),
            int(term["width"]),
        )
        if name in supplied or func in supplied:
            values = [float(v) for v in supplied.get(name, supplied.get(func, []))]
        elif func == "base_lin_vel":
            values = _rotate_to_body(_vector(observation, "base_quat", 4), _vector(observation, "base_lin_vel", 3))
        elif func == "base_ang_vel":
            values = _vector(observation, "base_ang_vel", 3)
        elif func == "projected_gravity":
            values = _rotate_to_body(_vector(observation, "base_quat", 4), [0.0, 0.0, -1.0])
        elif func == "generated_commands":
            values = [float(v) for v in (command if command is not None else [0.0] * width)]
        elif func in ("joint_pos_rel", "joint_vel_rel"):
            joints = term.get("joint_names") or list(contract.get("action_keys") or [])
            default = contract.get("default_joint_pos") or {}
            offsets = term.get("joint_offsets") or [float(default.get(j, 0.0)) for j in joints]
            suffix = "" if func == "joint_pos_rel" else ".vel"
            values = []
            for joint, offset in zip(joints, offsets, strict=False):
                key = joint_keys.get(joint, joint) + suffix
                if key not in observation:
                    raise DeployContractError(f"the observation has no {key!r} for the actor's {func} term")
                values.append(float(observation[key]) - (float(offset) if suffix == "" else 0.0))
        elif func == "last_action":
            values = [float(v) for v in last_action]
        elif func == "height_scan":
            base = _vector(observation, "base_pos", 3)
            offset = float((term.get("params") or {}).get("offset", 0.5))
            values = [base[2] - offset] * width
        else:
            raise DeployContractError(
                f"the actor reads an Isaac Lab {func!r} term ({name}) strands does not compute; pass its "
                f"{width} values as obs_terms={{{name!r}: [...]}}"
            )
        if len(values) != width:
            raise DeployContractError(f"term {name!r} produced {len(values)} values, the actor reads {width}")
        scale = term.get("scale")
        if isinstance(scale, int | float) and not isinstance(scale, bool):
            values = [v * float(scale) for v in values]
        clip = term.get("clip")
        if isinstance(clip, Sequence) and len(clip) == 2:
            values = [min(max(v, float(clip[0])), float(clip[1])) for v in values]
        out += values
    return out


def apply_action_contract(contract: Mapping[str, Any], raw: Sequence[float]) -> dict[str, float]:
    """Turn a raw actor output into named joint position targets, as Isaac Lab's action terms do.

    Per term: clip (when the term has a clip), then ``target = offset + scale * action``,
    bound to the term's joints by name - never by the deploying robot's position.

    Raises:
        DeployContractError: *raw* is not as wide as the contract's action keys.
    """
    keys = list(contract.get("action_keys") or [])
    if len(raw) != len(keys):
        raise DeployContractError(f"the actor emitted {len(raw)} values for {len(keys)} contract joints")
    targets: dict[str, float] = {}
    for term in contract.get("action_terms", []):
        start = int(term["start"])
        clip = term.get("clip")
        for i, joint in enumerate(term["joint_names"]):
            value = float(raw[start + i])
            if clip is not None:
                value = min(max(value, clip[i][0]), clip[i][1])
            targets[joint] = term["offset"][i] + term["scale"][i] * value
    return targets


__all__ = [
    "CONTRACT_VERSION",
    "DeployContractError",
    "ISAACLAB_BASE_VELOCITY_FRAME",
    "ISAACLAB_QUAT_ORDER",
    "POSITION_ACTION_TERMS",
    "apply_action_contract",
    "attach_env_cfg",
    "build_policy_obs",
    "complete_obs_layout",
    "contract_from_io_descriptors",
    "contract_problems",
]
