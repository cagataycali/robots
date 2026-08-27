"""Q54: the record form's fps must actually reach the session (backend half of the fix).

/api/record/open has always read `fps`; nothing sent it, so every dataset was stamped 30 while an
SO-101 captures nearer 4 -- and LeRobot derives each frame's timestamp from the declaration, so the
artifact claimed the motion happened ~7x faster than it did. This pins the wire contract the form
now depends on: the declared rate is honoured, a missing one still means 30, and junk cannot make
a session with a nonsense rate.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from strands_robots.dashboard.record_worker import RecordWorker


def _worker(**kw):
    class _Backend:
        """Only what the worker touches to report a session -- no arm, no frames."""

        cameras: dict = {}

        def leader_action(self):
            return {}

        def follower_apply(self, action):
            return {}

        def follower_observation(self):
            return {}

        def close(self):
            pass

    defaults = dict(
        dataset="cagatay/so101-pick",
        task="pick the cube",
        leader="so101-arm-2",
        follower="so101-arm-1",
        target_episodes=3,
        backend=_Backend(),
        recorder_factory=lambda **_: None,
        thumb_dir=pathlib.Path(tempfile.mkdtemp()),
        fps=30,
    )
    defaults.update(kw)
    return RecordWorker(**defaults)


@pytest.mark.parametrize("declared", [1, 4, 30, 60])
def test_the_declared_rate_is_what_the_session_reports(declared):
    w = _worker(fps=declared)
    assert w.fps == declared
    assert w.session()["fps"] == declared, "the panel reads this number back"


def test_the_worker_refuses_to_invent_a_rate():
    """The 30 lives in the ROUTE, deliberately: a worker with its own default would let a future
    caller forget the field and still produce a plausible-looking dataset."""
    import inspect

    sig = inspect.signature(RecordWorker.__init__)
    assert sig.parameters["fps"].default is inspect.Parameter.empty


def test_the_open_body_reading_matches_the_forms_contract():
    """The exact expression the route uses, pinned: `int(body.get("fps", 30) or 30)`.

    A 0 or an empty string means "no opinion" and must land on 30 rather than a rate that would
    divide by zero when timestamps are derived.
    """
    import inspect

    from strands_robots.dashboard import record_api

    src = inspect.getsource(record_api)
    assert 'fps=int(body.get("fps", 30) or 30)' in src
