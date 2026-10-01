"""Which policies a mesh peer can run, read from the registry for the agent.

The dashboard agent used to see a peer's ``execute`` action and a free-text
``policy_provider`` and nothing else, so asked to make a G1 walk it ran
``mock``. This module answers, per peer, from :mod:`strands_robots.registry`
and the peer's presence: the providers that apply to the robot the peer is,
the keys the wire carries for each, their types and bounds, and one line
saying what the provider does. ``fleet`` puts the answer on every row and the
system prompt tells the agent to pick from it.

Pure: no mesh, no model, no import of a provider module. What the wire
carries is what :func:`strands_robots.mesh.security.validate_command`
accepts for ``execute``; a key outside that set is refused on the robot host,
so it is not offered here either.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from strands_robots.locomotion_envelope import max_angular_velocity_rps, max_linear_velocity_mps

#: Providers bound to one embodiment: the robot names they load. A provider absent
#: here applies to any robot that can satisfy its checkpoint or server. Pinned
#: against the docs' ``EMBODIMENT_WITNESSES`` by a test so the two cannot drift.
EMBODIMENT_BOUND: dict[str, tuple[str, ...]] = {
    "wbc": ("unitree_g1",),
    "wbc_gait": ("unitree_g1",),
    "holosoma": ("unitree_g1",),
    "kimodo": ("unitree_g1",),
    "protomotions": ("unitree_g1",),
    "microduck": ("microduck",),
}

#: Providers a fleet row never offers: they need a server the dashboard cannot
#: start (a port on the robot host) or are removed in 0.7. ``remote`` stays: the
#: wire carries its ``server_address``.
NOT_OFFERED: frozenset[str] = frozenset({"cosmos3", "moveit2", "curobo", "kimodo", "protomotions", "flux3_action"})

#: One line per provider for the agent. A provider without a line gets the
#: registry description.
HINTS: dict[str, str] = {
    "mock": "sinusoidal test motion; proves the plumbing, never a task. Only when the operator asks for it",
    "wbc": "GR00T whole-body controller for the Unitree G1: balance and walk; target_velocity [vx, vy, wz] steers it",
    "wbc_gait": "GR00T gait-clock variant for the Unitree G1; target_velocity steers it",
    "holosoma": "Amazon FAR Holosoma locomotion for the Unitree G1 (fastsac or ppo); target_velocity steers it",
    "lerobot_local": "a LeRobot checkpoint (ACT, SmolVLA, pi0, ...) run in-process; needs pretrained_name_or_path",
    "remote": "a policy server elsewhere; needs server_address",
    "rl": "an RL actor exported by a trainer; needs model_path",
    "microduck": "the MicroDuck biped's walking controller",
}

#: The constructor keys the wire carries into ``create_policy`` (validated on the
#: robot host by ``mesh.security.validate_command``; ``walk`` and ``model_path``
#: reach the constructor through ``policy_config``).
WIRE_CONSTRUCTOR_KEYS: tuple[str, ...] = (
    "model_path",
    "pretrained_name_or_path",
    "policy_type",
    "server_address",
    "walk",
)

#: Provider config keys the wire's ``model_path`` stands for.
MODEL_PATH_ALIASES: tuple[str, ...] = ("model_path", "checkpoint")


def robot_of_peer(
    peer_id: str, peer: Mapping[str, Any] | None, managed: Iterable[Mapping[str, Any]] = ()
) -> str | None:
    """The registry robot a peer is, or None when nothing on the mesh says.

    A sim child ``<parent>__<robot>`` names its robot in its id; a sim parent with
    one robot names it in ``sim_robots``; a robot this dashboard spawned is in the
    managed table; a real arm's driver name is ``hw``.
    """
    if "__" in peer_id:
        suffix = peer_id.rsplit("__", 1)[1].strip()
        if suffix:
            return suffix
    presence = (peer or {}).get("presence") or {}
    sim_robots = presence.get("sim_robots")
    if isinstance(sim_robots, list) and len(sim_robots) == 1 and isinstance(sim_robots[0], str):
        return sim_robots[0]
    for row in managed:
        if row.get("peer_id") == peer_id and isinstance(row.get("robot_name"), str) and row["robot_name"]:
            return str(row["robot_name"])
    hw = presence.get("hw")
    if isinstance(hw, str) and hw.strip():
        return hw.strip()
    return None


def _velocity_bounds() -> dict[str, Any]:
    lin = max_linear_velocity_mps()
    ang = max_angular_velocity_rps()
    return {
        "type": "list[float]",
        "shape": "[vx, vy, wz]",
        "units": "m/s, m/s, rad/s",
        "bounds": {"vx": [-lin, lin], "vy": [-lin, lin], "wz": [-ang, ang]},
    }


def wire_kwargs_for(provider: str, info: Mapping[str, Any]) -> dict[str, Any]:
    """The kwargs the agent may send for this provider, typed, with what each is for."""
    keys = set(info.get("config_keys") or [])
    out: dict[str, Any] = {}
    if keys & set(MODEL_PATH_ALIASES):
        out["model_path"] = {"type": "str", "what": "a checkpoint directory on the robot host"}
    if "pretrained_name_or_path" in keys:
        out["pretrained_name_or_path"] = {
            "type": "str",
            "what": "a Hub id like lerobot/smolvla_base or a local checkpoint",
        }
    if "policy_type" in keys:
        out["policy_type"] = {"type": "str", "what": "the checkpoint's family: act, smolvla, pi0, ..."}
    if "server_address" in keys:
        out["server_address"] = {
            "type": "str",
            "what": "host:port of the policy server (allowlisted on the robot host)",
        }
    if "walk" in keys:
        out["walk"] = {"type": "bool", "what": "true = load the walk policy too; false = balance only"}
    if "target_velocity" in keys:
        out["target_velocity"] = {
            **_velocity_bounds(),
            "what": "the velocity command; per call, refused outside the bounds",
        }
    return out


def required_for(provider: str, info: Mapping[str, Any]) -> list[str]:
    """The keys without which ``create_policy`` refuses: the registry's ``requires`` plus a checkpoint for wbc."""
    required = [str(k) for k in (info.get("requires") or [])]
    keys = set(info.get("config_keys") or [])
    if "checkpoint" in keys and "model_path" not in required:
        required.append("model_path")
    return required


def policies_for_robot(robot: str | None, providers: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Provider rows that apply to ``robot`` (every unbound provider when the robot is unknown)."""
    rows: list[dict[str, Any]] = []
    for name, info in providers.items():
        if name in NOT_OFFERED:
            continue
        bound = EMBODIMENT_BOUND.get(name)
        if bound is not None and (robot is None or robot not in bound):
            continue
        rows.append(
            {
                "provider": name,
                "hint": HINTS.get(name) or str(info.get("description") or ""),
                "requires": required_for(name, info),
                "kwargs": wire_kwargs_for(name, info),
                "embodiments": list(bound) if bound else "any",
            }
        )
    # embodiment-bound first, the shipped GR00T controllers ahead of the rest: they are the answer to
    # "make it walk"; mock last
    first = ("wbc", "wbc_gait")
    rows.sort(
        key=lambda r: (
            r["embodiments"] == "any",
            r["provider"] == "mock",
            r["provider"] not in first,
            r["provider"],
        )
    )
    return rows


def _registry_providers() -> Mapping[str, Mapping[str, Any]]:
    from strands_robots.registry.policies import get_policy_provider, list_policy_providers

    return {name: info for name in list_policy_providers() if (info := get_policy_provider(name)) is not None}


def peer_policies(
    peer_id: str, peer: Mapping[str, Any] | None, managed: Iterable[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """The ``policies`` block for one fleet row: the robot, and the providers it can run."""
    robot = robot_of_peer(peer_id, peer, managed)
    return {"robot": robot, "can_run": policies_for_robot(robot, _registry_providers())}
