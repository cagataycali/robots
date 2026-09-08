"""Reading a streamed dataset in capture order is ``buffer_size=1``, not ``shuffle=False``.

``StreamingDatasetReader.open`` forwards ``shuffle`` to lerobot's
``StreamingLeRobotDataset``, where it decides only WHICH generator drives the
reordering: ``np.random.default_rng(seed)``, reseeded identically on every
exhaustion, or the dataset's own advancing one. lerobot documents it as
"whether to shuffle the dataset across exhaustions", and its ``__iter__``
reorders either way, at two levels - a shard sampled at random per frame, then a
reservoir buffer of ``buffer_size`` yielded from at a random index.

Two of the recipes a caller reads first - the ``StreamingDatasetReader`` class
docstring's "in-process eval / replay" Example and the streaming section of
``docs/recording.md`` - passed ``shuffle=False`` alone and commented it
"chronological for replay/eval". It is not: on a 60-frame recorded dataset that
recipe yields frame indices ``28, 55, 18, 24, 7, ...``, and passing
``shuffle=True`` alongside ``buffer_size=1`` reads in capture order anyway. So
the flag named for shuffling does not decide order, nothing reports a shuffled
read, and an eval or replay loop that followed the documented recipe consumed
its episode out of order while its own comment said otherwise.

The cells below pin both halves: the ordering contract, against a double shaped
like lerobot's iterator rather than like this module's description of it, and the
documented recipes, so a surface that claims capture order names the knob that
delivers it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import strands_robots.streaming_dataset as sd

REPO_ID = "local/capture_order"
FRAMES = 60


class _ReservoirShapedStreaming:
    """Stand-in reproducing the two reorderings lerobot's ``__iter__`` performs.

    Faithful to ``lerobot.datasets.streaming_dataset.StreamingLeRobotDataset``
    in exactly the respects that decide read order, so these cells measure the
    dependency's shape rather than this module's account of it:

    * ``shuffle`` selects the generator only - a fresh
      ``default_rng(seed)`` when False (identical on every exhaustion) versus an
      advancing one when True. Neither switches the reordering off.
    * shards are sampled at random per frame from
      ``min(hf_shards, max_num_shards)`` of them, so more than one shard
      interleaves.
    * frames pass through a reservoir of ``buffer_size``, yielded from at a
      random index, and whatever is left is shuffled once the shards are
      exhausted.

    ``hf_shards`` is a class attribute because the caller cannot reach it: the
    reader only forwards the knobs of :meth:`StreamingDatasetReader.open`.
    """

    hf_shards: ClassVar[int] = 1

    def __init__(self, repo_id: str, **kw: Any) -> None:
        self.repo_id = repo_id
        self.kw = kw
        self.buffer_size = int(kw.get("buffer_size", 1000))
        self.max_num_shards = int(kw.get("max_num_shards", 16))
        self.seed = int(kw.get("seed", 42))
        self.shuffle = bool(kw.get("shuffle", True))
        self.num_frames = FRAMES
        self.num_episodes = 1
        self.fps = 30

    def _shard(self, index: int, count: int) -> list[dict[str, int]]:
        """Frames of one shard, in capture order (HF shards round-robin)."""
        return [{"frame_index": i} for i in range(index, FRAMES, count)]

    def __iter__(self) -> Any:
        rng = np.random.default_rng(self.seed if not self.shuffle else self.seed + 1)
        count = min(self.hf_shards, self.max_num_shards)
        shards = {i: iter(self._shard(i, count)) for i in range(count)}
        buffer: list[dict[str, int]] = []
        while shards:
            key = int(rng.choice(list(shards)))
            frame = next(shards[key], None)
            if frame is None:
                del shards[key]
                continue
            if len(buffer) == self.buffer_size:
                slot = int(rng.integers(0, self.buffer_size))
                yield buffer[slot]
                buffer[slot] = frame
            else:
                buffer.append(frame)
        rng.shuffle(buffer)  # type: ignore[arg-type]
        yield from buffer


@pytest.fixture
def reservoir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the double through the override ``_get_streaming_cls`` honors."""
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _ReservoirShapedStreaming, raising=False)
    monkeypatch.setattr(_ReservoirShapedStreaming, "hf_shards", 1)


def _read_order(**kwargs: Any) -> list[int]:
    reader = sd.StreamingDatasetReader.open(REPO_ID, root="/tmp/unused", **kwargs)
    return [int(frame["frame_index"]) for frame in reader]


CAPTURE_ORDER = list(range(FRAMES))


def test_the_documented_recipe_reads_in_capture_order(reservoir: None) -> None:
    """``buffer_size=1`` + ``max_num_shards=1`` is what capture order needs."""
    assert _read_order(buffer_size=1, max_num_shards=1) == CAPTURE_ORDER


def test_shuffle_false_alone_does_not_read_in_capture_order(reservoir: None) -> None:
    """The recipe both docstring and docs used to carry reads reordered frames.

    Every frame is still delivered - the loss is the order, which is the whole
    point of a replay - so a caller has nothing to notice.
    """
    order = _read_order(shuffle=False)
    assert sorted(order) == CAPTURE_ORDER, "frames were lost, not merely reordered"
    assert order != CAPTURE_ORDER


def test_shuffle_true_with_the_recipe_still_reads_in_capture_order(reservoir: None) -> None:
    """Order does not depend on ``shuffle`` at all - the direct proof.

    Asking to shuffle and getting capture order anyway is what makes ``shuffle``
    the wrong knob to document as chronological.
    """
    assert _read_order(shuffle=True, buffer_size=1, max_num_shards=1) == CAPTURE_ORDER


def test_a_reservoir_of_one_is_not_enough_while_shards_interleave(
    monkeypatch: pytest.MonkeyPatch, reservoir: None
) -> None:
    """``max_num_shards=1`` earns its place in the recipe.

    A reservoir of one cannot reorder within a shard, but the shard sampled per
    frame is still random, so a multi-shard dataset interleaves without the
    second knob. A single-shard dataset cannot show this, which is why it is
    pinned against a double rather than against one recorded dataset.
    """
    monkeypatch.setattr(_ReservoirShapedStreaming, "hf_shards", 4)
    assert _read_order(buffer_size=1, max_num_shards=4) != CAPTURE_ORDER
    assert _read_order(buffer_size=1, max_num_shards=1) == CAPTURE_ORDER


# --- the documented recipes themselves ---------------------------------------

# Surfaces a caller reads to learn how to stream: the package, the guide, the
# runnable examples. tests_integ is deliberately excluded - its stream_dataset
# calls verify a schema, not an order, so they make no claim to keep honest.
_SURFACES = ("strands_robots", "docs", "examples")
# "def stream_dataset(" is the definition, not a use of the recipe.
_CALL = re.compile(r"(?<!def )(?:stream_dataset|StreamingDatasetReader\.open)\s*\(")
_CLAIM = re.compile(r"capture[ -]order|chronological", re.IGNORECASE)
_LINES_OF_LEAD_IN = 5


def _call_windows(text: str) -> list[str]:
    """Each streaming call's own argument text plus the lines leading into it."""
    windows = []
    for match in _CALL.finditer(text):
        depth, end = 0, match.end()
        for end in range(match.end() - 1, len(text)):
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            if depth == 0:
                break
        lead = text[: match.start()].splitlines()[-_LINES_OF_LEAD_IN:]
        windows.append("\n".join(lead) + text[match.start() : end + 1])
    return windows


def _documented_calls() -> list[tuple[str, str]]:
    root = Path(__file__).resolve().parent.parent
    found = []
    for surface in _SURFACES:
        for path in sorted((root / surface).rglob("*")):
            if path.suffix not in (".py", ".md"):
                continue
            for window in _call_windows(path.read_text(encoding="utf-8")):
                found.append((str(path.relative_to(root)), window))
    return found


def test_the_documented_surfaces_still_show_a_streaming_call() -> None:
    """Guard the rule below against silently scanning nothing."""
    assert len(_documented_calls()) >= 4


def test_every_surface_claiming_capture_order_names_the_knob_that_delivers_it() -> None:
    """A recipe may not promise capture order while passing only ``shuffle``.

    ``buffer_size=1`` is the knob that stops the reservoir reordering; a surface
    that claims capture order without it is documenting an order the reader does
    not deliver, which is exactly what the class docstring and
    ``docs/recording.md`` did.
    """
    offenders = [
        (where, window)
        for where, window in _documented_calls()
        if _CLAIM.search(window) and "buffer_size=1" not in window
    ]
    assert not offenders, "\n\n".join(f"{where}:\n{window}" for where, window in offenders)


def test_no_documented_surface_opts_out_of_shuffle_as_its_order_knob() -> None:
    """``shuffle=False`` may not stand in for the recipe, claim or no claim.

    Covers the surfaces the rule above cannot: an Example that passes
    ``shuffle=False`` and says nothing still teaches it as the way to read a
    recorded episode back, and that is how four surfaces came to carry a recipe
    that reorders. ``shuffle=True`` is untouched - a training example wants the
    shuffle and makes no claim on order.
    """
    offenders = [
        (where, window)
        for where, window in _documented_calls()
        if "shuffle=False" in window and "buffer_size=1" not in window
    ]
    assert not offenders, "\n\n".join(f"{where}:\n{window}" for where, window in offenders)
