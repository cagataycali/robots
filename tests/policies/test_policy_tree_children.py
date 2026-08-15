"""Tests for the policy-tree seam: ``Policy.children`` + ``iter_policy_tree``.

A runtime capability probe answers about the object it is handed. A *wrapper*
policy is a different object than the policy inside it, so a probe that
type-tests its argument reports the wrapped policy's capability as absent -
which is how a WBC policy inside a
:class:`~strands_robots.policies.composite.CompositePolicy` lost the MuJoCo
torque shim it needs, even though the physics the shim corrects is identical.

:attr:`~strands_robots.policies.base.Policy.children` is the declaration that
fixes that once for every probe, and
:func:`~strands_robots.policies.base.iter_policy_tree` is the walk over it.
These tests pin the contract the probes rely on: a leaf declares nothing, both
shipped wrappers declare what they drive, the walk reaches a policy nested
several wrappers deep, and a shared child or a cycle terminates instead of
double-reporting or hanging.
"""

from __future__ import annotations

from typing import Any

from strands_robots.policies import MockPolicy
from strands_robots.policies.base import Policy, iter_policy_tree
from strands_robots.policies.composite import CompositePolicy
from strands_robots.policies.persistent import PersistentPolicy


class _Leaf(Policy):
    """Minimal leaf policy that declares one named joint."""

    def __init__(self, joint: str) -> None:
        self.joint = joint

    @property
    def provider_name(self) -> str:
        return f"leaf-{self.joint}"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        return None

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict]:
        return [{self.joint: 0.0}]


class TestChildrenDeclaration:
    def test_a_leaf_policy_declares_no_children(self) -> None:
        assert MockPolicy().children == ()
        assert _Leaf("a").children == ()

    def test_composite_declares_both_children_lower_first(self) -> None:
        lower, upper = _Leaf("lo"), _Leaf("up")
        composite = CompositePolicy(lower=lower, upper=upper)
        assert composite.children == (lower, upper)

    def test_persistent_declares_the_policy_it_holds_warm(self) -> None:
        inner = _Leaf("inner")
        assert PersistentPolicy("mock", policy_object=inner).children == (inner,)


class TestIterPolicyTree:
    def test_a_leaf_yields_only_itself(self) -> None:
        leaf = _Leaf("solo")
        assert list(iter_policy_tree(leaf)) == [leaf]

    def test_the_root_is_yielded_before_its_children(self) -> None:
        lower, upper = _Leaf("lo"), _Leaf("up")
        composite = CompositePolicy(lower=lower, upper=upper)
        assert list(iter_policy_tree(composite)) == [composite, lower, upper]

    def test_a_probe_reaches_a_policy_nested_under_several_wrappers(self) -> None:
        # The shape that lost the torque shim: the policy a probe is looking for
        # sits two wrappers down, and neither wrapper is of its type.
        target = _Leaf("target")
        nested = PersistentPolicy("mock", policy_object=target)
        composite = CompositePolicy(lower=nested, upper=_Leaf("other"))
        found = next((p for p in iter_policy_tree(composite) if isinstance(p, _Leaf) and p.joint == "target"), None)
        assert found is target

    def test_a_child_shared_by_two_wrappers_is_yielded_once(self) -> None:
        shared = _Leaf("shared")
        composite = CompositePolicy(
            lower=PersistentPolicy("mock", policy_object=shared),
            upper=PersistentPolicy("mock", policy_object=shared),
        )
        walked = list(iter_policy_tree(composite))
        assert walked.count(shared) == 1

    def test_a_cycle_terminates_instead_of_recursing_forever(self) -> None:
        class _SelfReferencing(_Leaf):
            @property
            def children(self) -> tuple[Policy, ...]:
                return (self,)

        loop = _SelfReferencing("loop")
        assert list(iter_policy_tree(loop)) == [loop]
