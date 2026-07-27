"""Unit tests for ``RecordingMixin._verify_resume_schema``.

The schema-verification helper is a pure function of its arguments (it reads no
instance state), so these tests bind it to a dummy ``self`` and run without
mujoco or lerobot installed. They pin the #366 follow-up: resume() must reject a
scene whose schema diverges from the existing on-disk dataset rather than
deferring to a cryptic per-feature shape error on the next add_frame.

The dataset frame rate is part of that schema: a resume cannot change it, so a
differing ``fps`` must be refused rather than silently appending frames on the
dataset's timebase instead of the requested one.
"""

import pytest

from strands_robots.simulation.mujoco.recording import RecordingMixin

verify = RecordingMixin._verify_resume_schema


class _FakeRecorder:
    def __init__(self, features, fps=None, meta_fps=None):
        """Fake resumed recorder.

        Args:
            features: On-disk feature dict, or None to omit ``features``
                entirely (an unexpected LeRobot layout).
            fps: Value for the dataset's ``fps`` attribute, omitted when None.
            meta_fps: Value for ``dataset.meta.fps``, omitted when None.
        """
        attrs = {}
        if features is not None:
            attrs["features"] = features
        if fps is not None:
            attrs["fps"] = fps
        if meta_fps is not None:
            attrs["meta"] = type("_Meta", (), {"fps": meta_fps})()
        self.dataset = type("_DS", (), attrs)()


def _features(joint_names, cams, actions=None):
    """Build a minimal on-disk feature dict: cams maps name -> (h, w).

    When ``actions`` is given, an ``action`` feature is added carrying those
    column names, mirroring what ``DatasetRecorder`` writes for the actuator
    command vector.
    """
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": list(joint_names)},
    }
    if actions is not None:
        feats["action"] = {"dtype": "float32", "shape": (len(actions),), "names": list(actions)}
    for name, (h, w) in cams.items():
        feats[f"observation.images.{name}"] = {"dtype": "video", "shape": (3, h, w)}
    return feats


def test_resume_schema_matching_scene_passes():
    rec = _FakeRecorder(_features(["shoulder_pan", "elbow"], {"front": (480, 640)}))
    # No raise -> schema matches.
    verify(None, rec, ["shoulder_pan", "elbow"], ["front"], {"front": (480, 640)}, fps=30)


def test_resume_schema_extra_joint_raises():
    rec = _FakeRecorder(_features(["shoulder_pan"], {}))
    with pytest.raises(ValueError, match="observation.state joints differ"):
        verify(None, rec, ["shoulder_pan", "elbow"], [], {}, fps=30)


def test_resume_schema_camera_resolution_mismatch_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640)}))
    with pytest.raises(ValueError, match="resolution differs"):
        verify(None, rec, ["j"], ["front"], {"front": (256, 256)}, fps=30)


def test_resume_schema_new_camera_in_scene_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640)}))
    with pytest.raises(ValueError, match="not in the on-disk schema"):
        verify(None, rec, ["j"], ["front", "wrist"], {"front": (480, 640), "wrist": (480, 640)}, fps=30)


def test_resume_schema_dropped_camera_raises():
    rec = _FakeRecorder(_features(["j"], {"front": (480, 640), "wrist": (480, 640)}))
    with pytest.raises(ValueError, match="not in the current scene"):
        verify(None, rec, ["j"], ["front"], {"front": (480, 640)}, fps=30)


def test_resume_schema_no_features_skips_silently():
    """An unexpected LeRobot layout (no .features) must not block a valid resume."""
    rec = type("_R", (), {"dataset": type("_DS", (), {})()})()
    verify(None, rec, ["j"], [], {}, fps=30)  # no raise


def test_resume_schema_error_message_is_ascii():
    rec = _FakeRecorder(_features(["shoulder_pan"], {}))
    with pytest.raises(ValueError) as exc:
        verify(None, rec, ["shoulder_pan", "elbow"], [], {}, fps=30)
    str(exc.value).encode("ascii")  # raises if any non-ASCII glyph leaked


def test_resume_schema_action_columns_mismatch_raises():
    """A resumed dataset whose action columns diverge from the scene is rejected.

    Guards the actuator-command half of the schema check: joints/cameras can
    match while the action vector silently changed (e.g. a robot swapped for one
    with different actuators), which would otherwise only surface as a cryptic
    per-frame shape error on the next add_frame.
    """
    rec = _FakeRecorder(_features(["j"], {}, actions=["shoulder_pan", "elbow"]))
    with pytest.raises(ValueError, match="action columns differ"):
        verify(None, rec, ["j"], [], {}, action_names=["shoulder_pan", "wrist"], fps=30)


def test_resume_schema_matching_action_columns_passes():
    """Matching action columns must not raise when action_names is supplied."""
    rec = _FakeRecorder(_features(["j"], {}, actions=["shoulder_pan", "elbow"]))
    # No raise -> the action feature matches the scene's actuator columns.
    verify(None, rec, ["j"], [], {}, action_names=["shoulder_pan", "elbow"], fps=30)


def test_resume_schema_fps_mismatch_raises():
    """A resume at a rate the dataset cannot be re-created at is refused.

    ``LeRobotDataset.resume`` takes no ``fps``, so the dataset keeps the rate it
    was created at. Appending anyway timestamps the new frames at the on-disk
    rate although they were captured at the requested one, silently writing a
    wrong timebase into the dataset.
    """
    rec = _FakeRecorder(_features(["j"], {}), fps=30)
    with pytest.raises(ValueError, match="dataset fps differs: on-disk=30 vs requested=60"):
        verify(None, rec, ["j"], [], {}, fps=60)


def test_resume_schema_fps_error_names_the_rate_to_pass():
    """The refusal is actionable: it names the value that would append."""
    rec = _FakeRecorder(_features(["j"], {}), fps=30)
    with pytest.raises(ValueError, match=r"pass fps=30 to append at it"):
        verify(None, rec, ["j"], [], {}, fps=60)


def test_resume_schema_matching_fps_passes():
    rec = _FakeRecorder(_features(["j"], {}), fps=30)
    verify(None, rec, ["j"], [], {}, fps=30)  # no raise


def test_resume_schema_fps_read_from_dataset_metadata():
    """The rate is also honored when only ``dataset.meta.fps`` is exposed."""
    rec = _FakeRecorder(_features(["j"], {}), meta_fps=25)
    with pytest.raises(ValueError, match="on-disk=25 vs requested=30"):
        verify(None, rec, ["j"], [], {}, fps=30)


def test_resume_schema_fps_mismatch_raises_without_a_feature_dict():
    """The rate is metadata, so it is compared even with no ``features`` map.

    A LeRobot layout that does not expose ``features`` skips the joint/camera
    comparison, but the frame rate still comes from the dataset metadata and a
    mismatch there is enough to lose the timebase.
    """
    rec = _FakeRecorder(None, fps=30)
    with pytest.raises(ValueError, match="dataset fps differs"):
        verify(None, rec, ["j"], [], {}, fps=60)


def test_resume_schema_missing_fps_does_not_block_resume():
    """No reported rate -> no comparison, same best-effort posture as features."""
    rec = _FakeRecorder(_features(["j"], {}))
    verify(None, rec, ["j"], [], {}, fps=60)  # no raise


def test_resume_schema_fractional_disk_fps_does_not_block_resume():
    """A fractional on-disk rate has no whole-number equivalent to advise.

    ``start_recording`` only accepts a positive whole ``fps``, so there is no
    value the caller could pass to match a 29.97 fps dataset; refusing would
    dead-end the resume, so the comparison is skipped instead.
    """
    rec = _FakeRecorder(_features(["j"], {}), fps=29.97)
    verify(None, rec, ["j"], [], {}, fps=30)  # no raise


def test_resume_schema_boolean_disk_fps_is_not_read_as_a_rate():
    """``True`` is an int subclass but not a frame rate; it must not compare."""
    rec = _FakeRecorder(_features(["j"], {}), fps=True)
    verify(None, rec, ["j"], [], {}, fps=30)  # no raise


def test_resume_schema_reports_fps_and_scene_diffs_together():
    """Every divergence is listed in one refusal, not just the first found."""
    rec = _FakeRecorder(_features(["shoulder_pan"], {}), fps=30)
    with pytest.raises(ValueError) as exc:
        verify(None, rec, ["shoulder_pan", "elbow"], [], {}, fps=60)
    message = str(exc.value)
    assert "dataset fps differs" in message
    assert "observation.state joints differ" in message
    message.encode("ascii")  # raises if any non-ASCII glyph leaked
