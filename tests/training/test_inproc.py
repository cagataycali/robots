"""Tests for the in-process execution primitives shared by the training backends.

Covers :mod:`strands_robots.training._inproc`:

* :class:`_Tee` - the write-through tee that forwards every write/flush to BOTH
  a live stream and the per-run log file. Its load-bearing contract is
  *resilience*: a broken or closed stream on either side must never propagate an
  exception into the training loop ("never let logging break training"). Those
  swallow-and-continue branches were previously unpinned, so a refactor that
  narrowed the ``except`` or dropped the second stream would silently either
  crash a training run on a flaky file handle or lose the RUNNING-vs-learning
  verdict log.
* :func:`capture_to_file` - the context manager that tees stdout/stderr and
  installs a root-logger ``FileHandler`` for the run, and is a strict no-op when
  ``log_path is None`` (the non-rank-0 worker path, so only rank 0 writes the
  shared log).

:func:`elastic_launch_callable` - the shell-free ``torchrun`` replacement: it
drives torch's programmatic elastic agent to spawn ``nproc_per_node`` workers,
calls the given Python callable in each with arguments passed as Python objects
(no argv to assemble or inject into), and returns a ``{local_rank: worker_return}``
mapping. The tests spawn real single-node workers (``start_method="spawn"``, so
the worker fn is module-level and picklable) and assert both the return-mapping
shape and that each worker sees the ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``
that the agent injects - the contract HF ``TrainingArguments`` / lerobot
``accelerate`` read.

These tests intentionally assert only on stdout/stderr
tee-ing and the handler lifecycle, never on which *logging records* reach the
file: record visibility is governed by the ambient root-logger level and, under
pytest, by the log-capture plugin's handlers - global mutable state that would
make any record-level assertion order-dependent and flaky.
"""

from __future__ import annotations

import io
import logging

from strands_robots.training._inproc import (
    _Tee,
    call_callable,
    capture_to_file,
    elastic_launch_callable,
)


def _rank_env_worker(marker_dir: str) -> int:
    """elastic_launch worker (module-level so ``spawn`` can pickle it).

    Records the torch-injected distributed env (``LOCAL_RANK``/``WORLD_SIZE``) to
    a per-rank marker file and returns a value derived from this worker's global
    ``RANK`` so the caller can assert the ``{local_rank: worker_return}`` mapping.
    """
    import os
    from pathlib import Path

    rank = int(os.environ["RANK"])
    marker = Path(marker_dir) / f"rank_{rank}.txt"
    marker.write_text(f"{os.environ['LOCAL_RANK']},{os.environ['WORLD_SIZE']}", encoding="utf-8")
    return rank * 10


class _Boom:
    """A stream whose write/flush always raise - stands in for a closed or
    otherwise broken file handle mid-run."""

    def write(self, s: str) -> int:
        raise ValueError("stream is broken")

    def flush(self) -> None:
        raise ValueError("stream is broken")


class TestTeeWrite:
    """_Tee.write forwards to both streams and never propagates a stream error."""

    def test_forwards_to_both_streams(self):
        primary, secondary = io.StringIO(), io.StringIO()
        tee = _Tee(primary, secondary)
        tee.write("hello")
        assert primary.getvalue() == "hello"
        assert secondary.getvalue() == "hello"

    def test_returns_len_of_input(self):
        tee = _Tee(io.StringIO(), io.StringIO())
        # The io.TextIOBase contract: write() returns the number of characters
        # written. Callers (redirect_stdout consumers) rely on this.
        # write() forwards to both streams (a side-effect), so hoist each call
        # out of assert: assert bodies are discarded under ``python -O``.
        n_hello = tee.write("hello")
        assert n_hello == 5
        n_empty = tee.write("")
        assert n_empty == 0
        unicode_text = "\u2713 unicode"
        n_unicode = tee.write(unicode_text)
        assert n_unicode == len(unicode_text)

    def test_survives_primary_stream_failure(self):
        # The load-bearing property: a broken PRIMARY (live) stream must not stop
        # the SECONDARY (log file) from receiving the write, and must not raise.
        secondary = io.StringIO()
        tee = _Tee(_Boom(), secondary)
        n = tee.write("world")
        assert n == 5
        assert secondary.getvalue() == "world"

    def test_survives_secondary_stream_failure(self):
        # A broken log file must not stop the live stream or raise.
        primary = io.StringIO()
        tee = _Tee(primary, _Boom())
        n = tee.write("world")
        assert n == 5
        assert primary.getvalue() == "world"

    def test_survives_both_streams_failing(self):
        # Worst case: both sides broken. write() still returns len(s) and the
        # exception never reaches the training loop.
        tee = _Tee(_Boom(), _Boom())
        n = tee.write("xyz")
        assert n == 3


class TestTeeFlush:
    """_Tee.flush forwards to both streams and never propagates a stream error."""

    def test_forwards_flush_to_both(self):
        flushed = []

        class _Recorder(io.StringIO):
            def flush(self) -> None:
                flushed.append(id(self))

        primary, secondary = _Recorder(), _Recorder()
        _Tee(primary, secondary).flush()
        assert {id(primary), id(secondary)} == set(flushed)

    def test_survives_flush_failure_on_both(self):
        # A flush() on a broken stream must be swallowed - a failed flush on a
        # closed handle must not abort a training run.
        _Tee(_Boom(), _Boom()).flush()  # must not raise


class TestCaptureToFile:
    """capture_to_file tees stdout/stderr into the log and cleans up on exit."""

    def test_tees_stdout_to_file(self, tmp_path, capsys):
        log = tmp_path / "run.log"
        with capture_to_file(str(log)):
            print("STDOUT_LINE")
        assert "STDOUT_LINE" in log.read_text(encoding="utf-8")

    def test_tees_stderr_to_file(self, tmp_path):
        import sys

        log = tmp_path / "run.log"
        with capture_to_file(str(log)):
            sys.stderr.write("STDERR_LINE\n")
        assert "STDERR_LINE" in log.read_text(encoding="utf-8")

    def test_removes_handler_and_closes_on_exit(self, tmp_path):
        root = logging.getLogger()
        before = list(root.handlers)
        with capture_to_file(str(tmp_path / "run.log")):
            # Exactly one FileHandler added for the duration of the context.
            added = [h for h in root.handlers if h not in before]
            assert len(added) == 1
            assert isinstance(added[0], logging.FileHandler)
        # ...and removed again on exit (no handler leak across runs).
        assert list(root.handlers) == before

    def test_restores_streams_and_handler_on_exception(self, tmp_path):
        import sys

        root = logging.getLogger()
        before_handlers = list(root.handlers)
        before_out, before_err = sys.stdout, sys.stderr
        try:
            with capture_to_file(str(tmp_path / "run.log")):
                raise RuntimeError("training blew up")
        except RuntimeError:
            pass
        # Even on an exception inside the context, redirect_stdout/stderr unwind
        # and the FileHandler is removed - no leaked handler, no swapped streams.
        assert sys.stdout is before_out
        assert sys.stderr is before_err
        assert list(root.handlers) == before_handlers


class TestCaptureToFileNoop:
    """capture_to_file(None) is the strict no-op used by non-rank-0 workers."""

    def test_none_adds_no_handler(self):
        root = logging.getLogger()
        before = list(root.handlers)
        with capture_to_file(None):
            pass
        assert list(root.handlers) == before

    def test_none_does_not_redirect_stdout(self):
        import sys

        before_out, before_err = sys.stdout, sys.stderr
        with capture_to_file(None) as cap:
            assert sys.stdout is before_out
            assert sys.stderr is before_err
            assert cap.log_path is None

    def test_none_creates_no_file(self, tmp_path, monkeypatch):
        # A non-rank-0 worker must not create a stray log file in its cwd.
        monkeypatch.chdir(tmp_path)
        with capture_to_file(None):
            pass
        assert list(tmp_path.iterdir()) == []


class TestCallCallable:
    """call_callable runs fn in-process and returns its value, log_path optional."""

    def test_returns_value_without_log(self):
        # call_callable runs fn (a side-effect); hoist it out of assert so the
        # call is not discarded under ``python -O``.
        result = call_callable(lambda x, y: x + y, 2, 3)
        assert result == 5

    def test_forwards_args_and_kwargs_through_capture(self, tmp_path):
        log = tmp_path / "run.log"

        def fn(a, *, b):
            print(f"CALLED a={a} b={b}")
            return a * b

        result = call_callable(fn, 6, log_path=str(log), b=7)
        assert result == 42
        assert "CALLED a=6 b=7" in log.read_text(encoding="utf-8")


class TestElasticLaunchCallable:
    """elastic_launch_callable spawns workers and returns their rank-keyed results.

    This is the in-process, shell-free replacement for ``torchrun
    --nproc_per_node=N``: torch's elastic agent spawns the workers and the
    callable receives Python objects, so there is no command line to build or
    inject into. The tests run real single-node spawns (CPU-only; c10d needs no
    CUDA), which is why the worker is a module-level function.
    """

    def test_single_worker_returns_rank_mapping_and_injects_dist_env(self, tmp_path):
        # One worker: elastic_launch returns {local_rank: worker_return}. The
        # worker (global RANK 0) returns rank * 10 == 0.
        out = elastic_launch_callable(_rank_env_worker, nproc_per_node=1, fn_args=(str(tmp_path),))
        assert out == {0: 0}
        # The worker saw the torch-injected distributed env - no argv, no
        # torchrun binary: LOCAL_RANK=0 in a WORLD_SIZE=1 group.
        assert (tmp_path / "rank_0.txt").read_text(encoding="utf-8") == "0,1"

    def test_spawns_one_worker_per_nproc_with_distinct_ranks(self, tmp_path):
        # nproc_per_node=2 spawns exactly two workers with distinct global ranks
        # (0, 1), each keyed by its local rank in the returned mapping.
        out = elastic_launch_callable(_rank_env_worker, nproc_per_node=2, fn_args=(str(tmp_path),))
        assert out == {0: 0, 1: 10}
        # Both workers share WORLD_SIZE=2 and carry their own LOCAL_RANK.
        assert (tmp_path / "rank_0.txt").read_text(encoding="utf-8") == "0,2"
        assert (tmp_path / "rank_1.txt").read_text(encoding="utf-8") == "1,2"
