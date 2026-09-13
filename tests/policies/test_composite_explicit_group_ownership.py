"""An explicit CompositePolicy joint group is exclusive, on every tick alike.

``upper_joints`` names the joints the upper policy is authoritative for. A
defaulted ``lower_joints`` keeps every name the lower policy emits, so a lower
policy whose action space covers the whole robot also commands the names the
caller assigned to the upper policy. Which value reached the actuator was then
decided per tick by the upper policy's chunk contents::

    c = CompositePolicy(whole_body, arms, upper_joints=["armA", "armB"])

    c._merge_tick({"leg": 1.0, "armB": 9.9}, {"armA": 0.5, "armB": 0.5})
    # -> ValueError: both produced joint(s) ['armB']

    c._merge_tick({"leg": 1.0, "armB": 9.9}, {"armA": 0.5})
    # -> {'leg': 1.0, 'armB': 9.9, 'armA': 0.5}
    #                  ^^^^^^^^^^ armB is the upper policy's, driven by the lower
    #                             policy, status success, no warning

One configuration, two outcomes in different KINDS - crash or wrong actuator
command - chosen by whether the upper policy's chunk happened to cover that joint
on that tick. A chunk-emitting manipulation policy is silent on some of its own
joints on most ticks, so both outcomes occur within one rollout.

The mirror config is the same question with the seats swapped::

    c = CompositePolicy(legs_only, whole_body, lower_joints=LEGS_AND_WAIST)

    c._merge_tick({leg: -0.9 for leg in LEGS}, {j: 0.3 for j in ALL_29})
    # -> {'left_hip_pitch_joint': -0.9, ..., 'waist_yaw_joint': 0.3, ...}
    #                                        ^^^^^^^^^^^^^^^^^^^^^^^ the waist is
    #    in lower_joints, the caller gave it to the balance controller, and the
    #    manipulation policy drives it - status success, no warning

These pin that neither child may command into the other's explicit group at all,
so the answer is the same on every tick, plus the boundaries: precedence still
arbitrates two DEFAULTED groups (where the caller declared no owner), and the
disjoint splits every doc and example uses are untouched.
"""

import pytest

from strands_robots.policies import CompositePolicy
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS
from tests.policies.test_composite import StubPolicy, _run

G1_ARM_JOINTS = [j for j in WBC_G1_ALL_JOINTS if j not in set(WBC_G1_LEG_WAIST_JOINTS)]
G1_WAIST_JOINTS = [j for j in WBC_G1_LEG_WAIST_JOINTS if "waist" in j]
G1_LEG_JOINTS = [j for j in WBC_G1_LEG_WAIST_JOINTS if j not in set(G1_WAIST_JOINTS)]


def _waist_verdict(outcome: tuple[str, object]) -> str:
    """Who drives the G1 waist in this outcome - by value, not by claim."""
    kind, payload = outcome
    if kind == "refused":
        return "refused"
    values = {payload[j] for j in G1_WAIST_JOINTS}  # type: ignore[index]
    return {frozenset({-0.9}): "locomotion", frozenset({0.3}): "manipulation"}.get(
        frozenset(values), f"mixed {sorted(values)}"
    )


def _outcome(lower_tick: dict, upper_tick: dict, **groups) -> tuple[str, object]:
    """Run one tick through the public path; report the kind of answer it gave."""
    c = CompositePolicy(StubPolicy([dict(lower_tick)]), StubPolicy([dict(upper_tick)]), **groups)
    try:
        return ("merged", _run(c)[0])
    except ValueError as exc:
        return ("refused", str(exc))


class TestAnExplicitUpperGroupIsExclusive:
    """The lower policy may not command a joint ``upper_joints`` gave to the upper."""

    def test_the_answer_does_not_depend_on_what_upper_emitted_this_tick(self):
        """One config, two ticks: the same ownership question gets the same answer.

        The only difference between the ticks is whether the upper policy emitted
        ``armB`` - a property of its chunk, not of the ownership the caller
        declared. It must not decide who drives ``armB``.
        """
        lower_tick = {"leg": 1.0, "armB": 9.9}
        groups = {"upper_joints": ["armA", "armB"]}
        silent_on_armb = _outcome(lower_tick, {"armA": 0.5}, **groups)
        emits_armb = _outcome(lower_tick, {"armA": 0.5, "armB": 0.5}, **groups)
        assert silent_on_armb[0] == emits_armb[0], (
            f"the same composite answered {silent_on_armb[0]!r} when the upper policy was silent "
            f"on its own joint 'armB' and {emits_armb[0]!r} when it emitted it; the silent tick "
            f"returned {silent_on_armb[1]!r}"
        )

    def test_the_lower_policy_never_drives_a_joint_upper_explicitly_owns(self):
        """An upper-owned joint carries the upper policy's value or the tick is refused."""
        kind, payload = _outcome({"leg": 1.0, "armB": 9.9}, {"armA": 0.5}, upper_joints=["armA", "armB"])
        assert kind == "refused", (
            f"'armB' is assigned to the upper policy, which did not command it this tick, "
            f"yet the merged action drives it with the lower policy's value: {payload!r}"
        )

    def test_a_whole_body_lower_overlapping_the_arm_group_is_still_refused(self):
        """A lower policy that commands every arm joint every tick stays refused.

        The whole-body kinematic generators emit all 29 G1 joints, so they command
        the upper policy's whole group on every tick and therefore always overlap
        whatever the upper policy emits. That case was already refused - it is the
        tick where the two emitted sets happen NOT to overlap that went silently
        wrong - so this pins that the refusal survives.
        """
        lower = StubPolicy([{j: -0.9 for j in WBC_G1_ALL_JOINTS}], name="whole_body")
        upper = StubPolicy([{G1_ARM_JOINTS[0]: 0.3}], name="manipulation")
        c = CompositePolicy(lower, upper, upper_joints=G1_ARM_JOINTS)
        with pytest.raises(ValueError) as excinfo:
            _run(c)
        assert "upper_joints" in str(excinfo.value)

    def test_the_refusal_names_the_contested_joint_the_owner_and_a_remedy(self):
        """A refusal a caller can act on without reading the merge implementation."""
        kind, message = _outcome(
            {"leg": 1.0, "armB": 9.9},
            {"armA": 0.5},
            upper_joints=["armA", "armB"],
        )
        assert kind == "refused", message
        assert "armB" in message, message  # the contested joint
        assert "armA" not in message, message  # ... and not the uncontested one
        assert "stub" in message  # both children, by provider name
        assert "lower_joints" in message  # the remedy


class TestAnExplicitLowerGroupIsExclusive:
    """The upper policy may not command a joint ``lower_joints`` gave to the lower."""

    def test_who_drives_the_waist_does_not_depend_on_what_lower_emitted(self):
        """One config, two ticks: the same ownership question gets the same answer.

        The only difference between the ticks is whether the lower policy emitted
        the waist - a property of its chunk, not of the ownership the caller
        declared. It must not decide who drives the waist. Both ticks merged, so
        this direction never announced the disagreement in either kind: the waist
        simply carried the locomotion policy's value on one tick and the
        manipulation policy's on the other.
        """
        upper_tick = {j: 0.3 for j in WBC_G1_ALL_JOINTS}
        groups = {"lower_joints": WBC_G1_LEG_WAIST_JOINTS}
        silent_on_waist = _outcome({j: -0.9 for j in G1_LEG_JOINTS}, upper_tick, **groups)
        emits_waist = _outcome({j: -0.9 for j in WBC_G1_LEG_WAIST_JOINTS}, upper_tick, **groups)
        assert _waist_verdict(silent_on_waist) == _waist_verdict(emits_waist), (
            f"the same composite gave the waist to {_waist_verdict(silent_on_waist)} when the "
            f"locomotion policy was silent on its own waist joints and to "
            f"{_waist_verdict(emits_waist)} when it emitted them"
        )

    def test_the_upper_policy_never_drives_a_joint_lower_explicitly_owns(self):
        """A lower-owned joint carries the lower policy's value or the tick is refused.

        A whole-body manipulation policy emits all 29 G1 joints, so it commands the
        waist the caller assigned to the balance controller. A locomotion policy
        that drives the legs only leaves those names unclaimed, and the waist -
        part of how a humanoid stays upright - was written from the manipulation
        chunk instead.
        """
        kind, payload = _outcome(
            {j: -0.9 for j in G1_LEG_JOINTS},
            {j: 0.3 for j in WBC_G1_ALL_JOINTS},
            lower_joints=WBC_G1_LEG_WAIST_JOINTS,
        )
        assert kind == "refused", (
            f"the waist joints {G1_WAIST_JOINTS} are assigned to the lower policy, which did not "
            f"command them this tick, yet the merged action drives them with the upper policy's "
            f"value: {payload!r}"
        )

    def test_the_refusal_names_the_contested_joint_the_owner_and_a_remedy(self):
        """A refusal a caller can act on without reading the merge implementation."""
        kind, message = _outcome({"hip": 1.0}, {"hip": 9.9, "wrist": 2.0}, lower_joints=["hip", "knee"])
        assert kind == "refused", message
        assert "'hip'" in message, message  # the contested joint
        assert "knee" not in message, message  # ... and not the uncontested one
        assert "wrist" not in message, message  # ... nor the upper policy's own
        assert "stub" in message  # both children, by provider name
        assert "upper_joints" in message  # the remedy


class TestTheBoundariesAreUnchanged:
    """Only a child commanding into its sibling's EXPLICIT group is refused."""

    def test_two_defaulted_groups_are_still_arbitrated_by_lower_precedence(self):
        """With no group declared there is no owner to honour, so precedence stands.

        The upper policy's ``knee`` is still dropped in the lower policy's favour:
        that is the documented default, not a contested explicit assignment.
        """
        kind, merged = _outcome({"hip": 1.0, "knee": 1.0}, {"knee": 2.0, "wrist": 2.0})
        assert kind == "merged", merged
        assert merged == {"hip": 1.0, "knee": 1.0, "wrist": 2.0}

    def test_an_upper_policy_that_stays_out_of_the_lower_group_still_composes(self):
        """An explicit lower group only refuses an upper policy that reaches into it.

        The upper policy commands ``wrist`` alone, which ``lower_joints`` does not
        claim, so nothing is contested: the lower policy's stray ``wrist`` is
        filtered out by its own group (that is what declaring a group asks for)
        and the upper policy drives it.
        """
        kind, merged = _outcome({"hip": 1.0, "wrist": 9.9}, {"wrist": 2.0}, lower_joints=["hip"])
        assert kind == "merged", merged
        assert merged == {"hip": 1.0, "wrist": 2.0}

    def test_an_upper_group_the_lower_does_not_command_still_composes(self):
        """Correct use of the very config that had the hole keeps working."""
        kind, merged = _outcome({"leg": 1.0}, {"armA": 0.5, "armB": 0.5}, upper_joints=["armA", "armB"])
        assert kind == "merged", merged
        assert merged == {"leg": 1.0, "armA": 0.5, "armB": 0.5}

    def test_the_shipped_locomotion_split_is_unaffected(self):
        """A locomotion policy emits its own group only, so nothing is contested.

        ``WBCPolicy`` emits its 15 leg+waist joints and none of the 14 arm joints,
        so the pairing the class exists for does not reach the new refusal even
        with ``lower_joints`` left default.
        """
        lower = StubPolicy([{j: -0.9 for j in WBC_G1_LEG_WAIST_JOINTS}], name="wbc")
        upper = StubPolicy([{j: 0.3 for j in G1_ARM_JOINTS}], name="manipulation")
        c = CompositePolicy(lower, upper, upper_joints=G1_ARM_JOINTS)
        merged = _run(c)[0]
        assert set(merged) == set(WBC_G1_ALL_JOINTS)
        assert all(merged[j] == -0.9 for j in WBC_G1_LEG_WAIST_JOINTS)
        assert all(merged[j] == 0.3 for j in G1_ARM_JOINTS)
