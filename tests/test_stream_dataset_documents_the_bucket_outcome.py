"""``stream_dataset`` must not document a refusal it cannot raise.

``repo_type="bucket"`` reaches lerobot's ``StreamingLeRobotDataset``
unconditionally - the forwarding is pinned by
``tests/test_streaming_dataset.py::test_open_forwards_every_knob_to_the_constructor``,
because a dropped ``repo_type`` would silently read the versioned dataset
namespace instead of the requested bucket. There is therefore no version probe
left to refuse a below-floor lerobot, and the outcome on one is lerobot's own
``TypeError`` for an unexpected keyword.

The facade's docstring is the only place a caller learns which failure to
handle, so naming ``RuntimeError`` there sends them to write an ``except``
clause that can never run while the real ``TypeError`` escapes it.
"""

import pytest

import strands_robots.streaming_dataset as sd
from strands_robots import Simulation


class _BelowFloorStreaming:
    """A pre-``BUCKET_STREAMING_MIN_LEROBOT`` ``StreamingLeRobotDataset``.

    Its constructor predates ``repo_type``, which is exactly what a below-floor
    lerobot presents to the forwarding call.
    """

    def __init__(
        self,
        repo_id,
        root=None,
        tolerance_s=1e-4,
        streaming=True,
        buffer_size=1000,
        max_num_shards=16,
        seed=42,
        shuffle=True,
        return_uint8=True,
    ):
        self.repo_id = repo_id
        self.fps = 30
        self.num_frames = 1
        self.num_episodes = 1


def test_a_below_floor_lerobot_surfaces_its_own_typeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bucket request is forwarded, so the constructor rejects the keyword."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _BelowFloorStreaming, raising=False)

    with pytest.raises(TypeError, match="repo_type"):
        sd.StreamingDatasetReader.open("org/ds", repo_type="bucket", validate_deltas=False)


def test_the_facade_does_not_promise_a_refusal_it_cannot_raise() -> None:
    """The documented ``repo_type`` outcome names the exception callers can catch."""
    clause = (Simulation.stream_dataset.__doc__ or "").split("``repo_type``", 1)[-1]

    assert "RuntimeError" not in clause, (
        "stream_dataset documents RuntimeError for repo_type, but the keyword is "
        "forwarded unconditionally and a below-floor lerobot raises TypeError: "
        "callers would guard the wrong exception"
    )
    assert "TypeError" in clause, "the documented repo_type outcome must name the exception that is raised"
