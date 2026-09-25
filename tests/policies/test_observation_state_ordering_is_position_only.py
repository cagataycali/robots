# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An observation-derived joint-state ordering is position-only, in every provider.

A ``Policy`` whose caller declared no ``robot_state_keys`` has to infer this
step's state ordering from the observation, and the only ordering available is
the observation's own insertion order. Every sim backend writes a velocity
companion beside each joint position (``obs[jnt]`` then ``obs[f"{jnt}.vel"]``,
which ``simulation/mujoco/rendering.py`` calls an "Additive key ... existing
position-only" consumer contract), so that order alternates
``[pos0, vel0, pos1, vel1, ...]``. Taken as the state vector it puts a velocity
in every other slot and truncates the trailing joints away - a wrong state
vector with no error raised.

The LeRobot provider filtered here and stated exactly that harm; the Cosmos 3
provider inferred its ordering the same way and did not, so one 7-joint request
carried ``[joint1, joint1.vel, joint2, joint2.vel, joint3, joint3.vel, joint4]``
while the other carried ``joint1..joint7``, from the same observation. Two
providers conforming to the same ``Policy`` contract must read one observation
as one state vector, so the rule is shared
(:mod:`strands_robots.policies._state_keys`) rather than re-derived per
provider - the same reason the client-side RNG reseed is shared.

This module grades three things:

* the rule's own contract - a paired ``.vel`` is dropped, an unpaired one is
  kept (LeKiwi declares body-frame base velocities with no position companion),
  order is preserved;
* structurally, that every ``robot_state_keys or <inferred ordering>`` fallback
  in the tree routes its inferred side through the rule, so a third provider
  written the same way is held to it on arrival, and that the rule is defined
  once and in the shared leaf rather than inside a provider;
* behaviourally, that both providers read one sim-shaped observation as the
  same positions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

import strands_robots.policies as policies_pkg

_POLICIES_DIR = Path(policies_pkg.__file__).parent
_PACKAGE_DIR = _POLICIES_DIR.parent
_RULE_NAME = "drop_velocity_siblings"
# The suffix the sim backends append to a joint's additive velocity companion,
# spelled as the producer writes it (``simulation/mujoco/rendering.py``).
_VEL = ".vel"
#: The parameter both providers use to hand a resolver the observation's own
#: scalar keys. A resolver that renames it drops out of the scan, which is what
#: the roster guard below is for.
_OBSERVATION_ORDERING_PARAMS = frozenset({"scalar_keys"})

# A 7-DOF arm + gripper observation shaped exactly as the MuJoCo backend emits
# it: every joint position immediately followed by its additive `.vel` companion.
_SIM_SCALAR_KEYS = [key for j in range(1, 8) for key in (f"joint{j}", f"joint{j}{_VEL}")] + [
    "finger_joint1",
    f"finger_joint1{_VEL}",
]
_SIM_POSITIONS = [f"joint{j}" for j in range(1, 8)] + ["finger_joint1"]


def _sim_observation() -> dict[str, float]:
    """A sim-shaped scalar observation; velocities carry an unmistakable sentinel."""
    obs: dict[str, float] = {}
    for key in _SIM_SCALAR_KEYS:
        obs[key] = -100.0 if key.endswith(_VEL) else round(0.1 * (len(obs) // 2 + 1), 3)
    return obs


def _inferred_ordering_fallbacks() -> dict[str, str]:
    """Map ``relpath:lineno`` -> the alternative expression, for every fallback.

    A fallback is a ``<...robot_state_keys...> or <alternative>`` whose
    alternative INFERS an ordering (it contains a comprehension over the
    observation) rather than merely testing membership: ``key in
    self.robot_state_keys or key == "task"`` asks a boolean question about one
    key and is not an ordering, so it is not graded.
    """
    found: dict[str, str] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
                continue
            if "robot_state_keys" not in ast.unparse(node.values[0]):
                continue
            alternative = ast.unparse(
                node.values[1] if len(node.values) == 2 else ast.BoolOp(op=ast.Or(), values=node.values[1:])
            )
            infers_ordering = any(
                isinstance(inner, ast.ListComp | ast.SetComp | ast.GeneratorExp)
                for value in node.values[1:]
                for inner in ast.walk(value)
            )
            if infers_ordering:
                rel = source_file.relative_to(_PACKAGE_DIR)
                found[f"{rel}:{node.lineno}"] = alternative
    return found


def _is_observation_key_comprehension(node: ast.AST) -> bool:
    """True for a comprehension yielding a mapping's own keys in its order.

    ``[k for k, v in observation.items() if ...]`` infers an ordering; a dict
    comprehension rebuilds a mapping and ``[f"joint_{i}" for i in range(n)]``
    invents names the observation never carried, so neither is one.
    """
    if not isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp):
        return False
    if not node.generators:
        return False
    iterated = node.generators[0].iter
    if not (isinstance(iterated, ast.Call) and isinstance(iterated.func, ast.Attribute)):
        return False
    if iterated.func.attr not in {"items", "keys"}:
        return False
    target = node.generators[0].target
    key_name = (
        getattr(target.elts[0], "id", None)
        if isinstance(target, ast.Tuple) and target.elts
        else getattr(target, "id", None)
    )
    return isinstance(node.elt, ast.Name) and node.elt.id == key_name


def _resolver_inferred_returns() -> dict[str, str]:
    """Map ``relpath:lineno`` -> returned expression, for a resolver's inferred branches.

    The same two-branch fallback as ``robot_state_keys or <inferred ordering>``,
    spelled as statements instead of an expression: a function that hands back
    the declared ordering on one branch and an ordering it derived from the
    observation on another. Both providers resolve their ordering this way, so
    a scan that only reads the ``or`` form grades neither.

    Only the observation-derived branches are returned. A branch that hands
    back the declared keys is the declared side, and an ordering that never
    came from the observation - MoveIt2's planner roster, a positional
    ``joint_<i>`` label - is not an inferred ordering and carries no velocity
    companions to drop.
    """
    found: dict[str, str] = {}
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            returns = [stmt for stmt in ast.walk(node) if isinstance(stmt, ast.Return) and stmt.value is not None]
            if len(returns) < 2:
                continue
            # One hop of local binding, so `order = drop_velocity_siblings(...)`
            # followed by `return order` reads as the call it returns.
            bound: dict[str, ast.expr] = {
                stmt.targets[0].id: stmt.value
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            }
            parameters = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            resolved: list[tuple[ast.Return, ast.expr, str]] = []
            for stmt in returns:
                value: ast.expr | None = stmt.value
                if value is None:
                    continue
                if isinstance(value, ast.Name) and value.id in bound:
                    value = bound[value.id]
                resolved.append((stmt, value, ast.unparse(value)))
            # Not a resolver unless one branch hands back the declared ordering.
            if not any("robot_state_keys" in text for _, _, text in resolved):
                continue
            for stmt, value, text in resolved:
                if "robot_state_keys" in text:
                    continue
                derived_here = any(_is_observation_key_comprehension(inner) for inner in ast.walk(value))
                derived_by_caller = any(
                    isinstance(inner, ast.Name) and inner.id in parameters and inner.id in _OBSERVATION_ORDERING_PARAMS
                    for inner in ast.walk(value)
                )
                if derived_here or derived_by_caller:
                    found[f"{source_file.relative_to(_PACKAGE_DIR)}:{stmt.lineno}"] = text
    return found


def _inferred_state_orderings() -> dict[str, str]:
    """Every inferred ordering in the tree, in either spelling."""
    return {**_inferred_ordering_fallbacks(), **_resolver_inferred_returns()}


def _calls_the_shared_rule(expression: str) -> bool:
    """True if ``expression`` calls the shared rule, not a provider-local copy.

    Matched as a whole token: a re-derived ``_drop_velocity_siblings`` contains
    the shared name as a substring, and two providers each calling their own
    copy is the divergence this module exists to prevent.
    """
    return re.search(rf"(?<![\w.]){_RULE_NAME}\s*\(", expression) is not None


def _rule_definitions() -> list[str]:
    """Every ``def drop_velocity_siblings`` in the package, as ``relpath:lineno``."""
    sites: list[str] = []
    for source_file in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.endswith(_RULE_NAME):
                sites.append(f"{source_file.relative_to(_PACKAGE_DIR)}:{node.lineno}")
    return sites


# --- the rule's own contract ---------------------------------------------


def test_a_paired_velocity_sibling_is_dropped_and_order_is_preserved() -> None:
    """A sim ordering reduces to its joint positions, in the order they appeared."""
    from strands_robots.policies._state_keys import drop_velocity_siblings

    assert drop_velocity_siblings(_SIM_SCALAR_KEYS) == _SIM_POSITIONS


def test_an_unpaired_velocity_key_is_kept() -> None:
    """LeKiwi declares body-frame base velocities with no position companion.

    Dropping by suffix alone would empty that embodiment's state instead, so
    pairing is decided per key.
    """
    from strands_robots.policies._state_keys import drop_velocity_siblings

    base_velocities = ["x.vel", "y.vel", "theta.vel"]
    assert drop_velocity_siblings(base_velocities) == base_velocities


def test_a_declared_ordering_naming_a_velocity_is_only_filtered_when_paired() -> None:
    """The rule reads pairing, not intent: an operator's ordering is the caller's to pass or not.

    ``elbow.vel`` beside no ``elbow`` states the model's input and survives;
    beside ``elbow`` it is the additive companion and does not. Providers apply
    this to an INFERRED ordering only - a declared ``robot_state_keys`` never
    reaches it.
    """
    from strands_robots.policies._state_keys import drop_velocity_siblings

    assert drop_velocity_siblings(["elbow.vel", "wrist"]) == ["elbow.vel", "wrist"]
    assert drop_velocity_siblings(["elbow", "elbow.vel", "wrist"]) == ["elbow", "wrist"]


def test_an_empty_ordering_stays_empty() -> None:
    """No keys in, no keys out - the rule adds nothing."""
    from strands_robots.policies._state_keys import drop_velocity_siblings

    assert drop_velocity_siblings([]) == []


# --- structural: one owner, and every fallback routes through it ---------


def test_the_rule_is_defined_once_and_in_the_shared_leaf() -> None:
    """One owner, outside any provider, so no provider can re-derive a different rule."""
    owners = [site.split(":")[0] for site in _rule_definitions()]
    assert owners == ["policies/_state_keys.py"], owners


def test_every_inferred_state_ordering_routes_through_the_rule() -> None:
    """An ordering inferred from the observation must infer positions only.

    Graded in either spelling - the inline ``robot_state_keys or <inferred>``
    and a resolver that returns the two branches as statements - so moving one
    into the other cannot move it out of the rule.
    """
    offenders = {
        site: alternative
        for site, alternative in _inferred_state_orderings().items()
        if not _calls_the_shared_rule(alternative)
    }
    assert not offenders, (
        "these fall back to an ordering inferred from the observation without dropping the "
        f"additive velocity companions, so a sim observation is read as "
        f"[pos, vel, pos, vel, ...]: {offenders}"
    )


def test_the_structural_scan_found_the_fallbacks_it_grades() -> None:
    """Guard: a scan that reaches nothing would report every tree clean.

    Both providers are named, not just one: the scan used to read only the
    inline ``or`` spelling, which the LeRobot resolver never used, so it graded
    a single site and went empty the moment that site was rewritten as the
    statements the other provider already used.
    """
    orderings = _inferred_state_orderings()
    assert orderings, "no declared-or-inferred state-ordering fallback was found at all"
    for provider in ("policies/cosmos3/policy.py", "policies/lerobot_local/policy.py"):
        assert any(site.startswith(provider) for site in orderings), (provider, orderings)


def test_a_membership_test_against_the_declared_keys_is_not_graded() -> None:
    """``key in self.robot_state_keys or key == "task"`` is a boolean, not an ordering.

    Grading it would demand a velocity filter from a camera-partitioning loop
    that never builds a state vector.
    """
    assert not any(site.startswith("policies/lerobot_async/") for site in _inferred_ordering_fallbacks())


# --- behavioural: both providers read one observation the same way -------


def test_cosmos3_infers_positions_only_from_a_sim_observation() -> None:
    """With no declared keys, the 7-joint request carries positions, not velocities."""
    from strands_robots.policies.cosmos3.policy import Cosmos3Policy

    policy = Cosmos3Policy.__new__(Cosmos3Policy)
    policy.robot_state_keys = []
    out: dict[str, Any] = {}
    Cosmos3Policy._attach_joint_state(policy, _sim_observation(), out)

    joints = out["observation/joint_position"].reshape(-1).tolist()
    assert joints == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), joints
    assert not [v for v in joints if v < -50.0], f"velocity values reached the request: {joints}"


def test_both_providers_read_one_sim_observation_as_the_same_positions() -> None:
    """Parity: two providers inferring an ordering must infer the same one."""
    pytest.importorskip("torch")
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    lerobot_policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
    lerobot_policy.robot_state_keys = []
    lerobot_policy.strict_keys = False
    observation = _sim_observation()
    lerobot_order = LerobotLocalPolicy._resolve_state_order(lerobot_policy, observation, list(observation))

    from strands_robots.policies.cosmos3.policy import Cosmos3Policy

    cosmos_policy = Cosmos3Policy.__new__(Cosmos3Policy)
    cosmos_policy.robot_state_keys = []
    out: dict[str, Any] = {}
    Cosmos3Policy._attach_joint_state(cosmos_policy, observation, out)
    cosmos_joints = out["observation/joint_position"].reshape(-1).tolist()

    assert lerobot_order == _SIM_POSITIONS, lerobot_order
    assert cosmos_joints == pytest.approx([observation[key] for key in lerobot_order[:7]]), (
        cosmos_joints,
        lerobot_order,
    )


def test_the_shared_constant_is_the_producer_s_suffix() -> None:
    """The rule and this module's fixture must name one suffix, not two."""
    from strands_robots.policies._state_keys import VELOCITY_SUFFIX

    assert VELOCITY_SUFFIX == _VEL


def test_a_provider_local_copy_does_not_satisfy_the_routing_rule() -> None:
    """A re-derived ``_drop_velocity_siblings`` is not the shared rule.

    Two providers each filtering with their own copy is how the orderings
    diverged, so the routing check reads the shared name as a whole token.
    """
    assert _calls_the_shared_rule("drop_velocity_siblings(keys)")
    assert not _calls_the_shared_rule("_drop_velocity_siblings(keys)")
    assert not _calls_the_shared_rule("self._drop_velocity_siblings(keys)")
