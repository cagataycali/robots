"""A requested camera that never opened must be visible BEFORE collection.

The record worker derives ``camera_keys`` from the follower's first observation,
so a camera the machine refuses to open is merely absent — lerobot then builds
the dataset schema from what is present, every episode reports success, and the
dataset has no image channel to train on. This file pins the verdict that says
so, and pins that it stays silent when there is nothing to report (a false
camera warning on a healthy rig would teach the operator to ignore it).
"""

from strands_robots.dashboard.record_worker import camera_verdict


class TestSilenceWhenThereIsNothingToSay:
    def test_all_requested_cameras_present_is_silent(self):
        assert camera_verdict({"top": {}, "wrist": {}}, ["top", "wrist"]) is None

    def test_no_cameras_requested_and_none_present_is_silent(self):
        assert camera_verdict({}, []) is None
        assert camera_verdict(None, None) is None

    def test_an_extra_present_camera_is_not_a_problem(self):
        """Only the requested set is a promise; a bonus channel breaks nothing."""
        assert camera_verdict({"top": {}}, ["top", "wrist"]) is None


class TestTheVerdictNamesTheConsequence:
    def test_every_camera_missing_says_the_dataset_has_no_image_channel(self):
        v = camera_verdict({"top": {}, "wrist": {}}, [])
        assert v is not None
        assert v["missing"] == ["top", "wrist"]
        assert v["present"] == []
        assert "NO image channel" in v["message"]
        assert "cannot train a visual policy" in v["message"]

    def test_one_camera_missing_names_only_that_one(self):
        v = camera_verdict({"top": {}, "wrist": {}}, ["top"])
        assert v is not None
        assert v["missing"] == ["wrist"]
        assert v["present"] == ["top"]
        assert "wrist" in v["message"] and "missing those image channels" in v["message"]
        assert "NO image channel" not in v["message"], "one live camera is not zero"

    def test_the_message_points_at_where_the_reason_lives(self):
        v = camera_verdict({"top": {}}, [])
        assert v is not None
        assert "log" in v["message"], "the operator needs somewhere to look"
        assert "daemon" in v["message"], "the macOS TCC trap is the common cause here"

    def test_counts_are_requested_not_missing(self):
        v = camera_verdict({"a": {}, "b": {}, "c": {}}, ["a"])
        assert v is not None
        assert "2 of 3" in v["message"]


class TestTheSessionCarriesIt:
    """The verdict is only useful if the screen that collects can see it."""

    def test_session_exposes_the_notice_and_a_plain_backend_stays_none(self):
        from tests.test_dashboard_record_worker import make_worker

        worker, backend, _rec, _clock = make_worker()
        session = worker.session()
        assert "camera_notice" in session, "the record screen cannot warn about what it is not told"
        assert session["camera_notice"] is None, "a backend without the attribute must not break"

        # A backend that DID measure a missing camera surfaces it unchanged.
        backend.camera_notice = camera_verdict({"top": {}, "wrist": {}}, ["top"])
        assert worker.session()["camera_notice"]["missing"] == ["wrist"]
