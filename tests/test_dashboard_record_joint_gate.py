"""The record gate refuses an arm that cannot say where it is (record_joints).

Measured on the live rig the day this was written: both real arms published zero joints (one
port-contended, one uncalibrated), every existing gate passed, and the failure surfaced as
"500 could not open the arms: <exception>" — a raw traceback for a fault the dashboard had already
diagnosed. These tests pin the refusal AND, more importantly, every case where it must stay quiet.
"""

from __future__ import annotations

from strands_robots.dashboard import record_joints

NOW = 1_000_000.0


def peer(joints, *, age=2.0):
    p = {"peer_id": "so101-leader", "last_seen": NOW - age}
    p["state"] = {"peer_id": "so101-leader", "t": NOW} if joints is None else {"joints": joints}
    return p


class TestItRefusesWhenThereIsRealEvidence:
    def test_a_fresh_snapshot_with_no_joints_is_refused_for_the_follower(self):
        r = record_joints.refusal(role="follower", peer_id="so101-follower", peer=peer(None), now=NOW)
        assert "so101-follower" in r and "NO joint positions" in r
        assert "observations" in r, "the follower's joints are the dataset's observations"
        assert "2s old" in r, "how old the evidence is, so the operator can judge it"

    def test_the_leader_is_gated_too_and_named_as_actions(self):
        r = record_joints.refusal(role="leader", peer_id="so101-leader", peer=peer({}), now=NOW)
        assert "actions" in r, "the leader's joints are the dataset's actions"

    def test_the_classified_reason_and_remedy_travel_with_it(self):
        r = record_joints.refusal(
            role="follower",
            peer_id="arm",
            peer=peer(None),
            now=NOW,
            problem={"headline": "this board has no calibration", "remedy": "Respawn it as leader_arm."},
        )
        assert "this board has no calibration." in r
        assert "Respawn it as leader_arm." in r
        assert "devices > logs" not in r, "the generic fallback must not tag along behind a real reason"

    def test_without_a_classified_reason_it_points_at_the_log(self):
        r = record_joints.refusal(role="follower", peer_id="arm", peer=peer(None), now=NOW)
        assert "devices > logs" in r


class TestItStaysQuietWithoutEvidence:
    """Each of these would block a legitimate recording, which is worse than the 500 it replaces."""

    def test_joints_present_proceeds(self):
        assert (
            record_joints.refusal(role="follower", peer_id="arm", peer=peer({"shoulder_pan.pos": 1.0}), now=NOW) is None
        )

    def test_no_snapshot_at_all_proceeds(self):
        for p in (None, {}, "not a mapping", 7):
            assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None

    def test_a_peer_with_no_state_block_proceeds(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer={"last_seen": NOW - 1}, now=NOW) is None

    def test_a_stale_snapshot_is_not_evidence_about_now(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer(None, age=31.0), now=NOW) is None
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer(None, age=29.0), now=NOW) is not None

    def test_an_undateable_reading_is_not_evidence_either(self):
        p = peer(None)
        p.pop("last_seen")
        assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None
        p["last_seen"] = "yesterday"
        assert record_joints.refusal(role="follower", peer_id="arm", peer=p, now=NOW) is None

    def test_a_joints_shape_we_do_not_understand_is_not_absence(self):
        assert record_joints.refusal(role="follower", peer_id="arm", peer=peer("six"), now=NOW) is None
