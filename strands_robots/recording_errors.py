"""The errors a dataset recording raises, below the writer and its drivers both.

:class:`RecordingFrameError` is raised by
:meth:`strands_robots.dataset_recorder.DatasetRecorder.add_frame` in the ``app``
layer and caught by the rollout drivers in
:mod:`strands_robots.simulation.policy_runner` one layer under it, which is the
only reason the simulation layer imported an ``app`` module at all. An error type
that lives with whatever raises it is a contract every catcher has to reach
upward for, so this one sits in ``core`` beside the rest of the vocabulary two
layers have to agree on (:mod:`strands_robots.refusal_codes`,
:mod:`strands_robots.episode_labels`).

Nothing internal is imported here, and nothing outside the standard library: a
driver that needs to name the error must not pay for the recorder's numpy and
LeRobot imports to do it.
"""

from __future__ import annotations


class RecordingFrameError(RuntimeError):
    """A frame the dataset recorder could not write, in fail-fast mode.

    Raised by :meth:`strands_robots.dataset_recorder.DatasetRecorder.add_frame`
    when the underlying ``LeRobotDataset`` write fails and the recorder was
    constructed with ``strict=True`` (the default). The frame is already gone at
    that point, so the episode on disk is shorter than the rollout that produced
    it and every surviving frame is re-timestamped from the declared ``fps`` -
    the caller has to be told.

    A distinct type, rather than the underlying error, so a rollout driver can
    tell a lost recording frame apart from a failure in a caller's telemetry
    hook. The drivers deliberately tolerate a few consecutive telemetry
    failures; granting that tolerance to a lost recording frame truncates the
    dataset while the rollout still reports success. The originating error is
    chained and its text preserved.
    """
