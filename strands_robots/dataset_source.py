"""The dataset a ``repo_id`` names: where it lives, and the episode a reader opens.

One id addresses a LeRobot dataset, and two questions have to be answered the
same way by everyone who touches it: which directory on disk that id means, and
where an episode's frames start and end inside it. Four layers ask them -- the
recorder session writes through :func:`resolve_dataset_dir`, the three
simulation backends stash the same directory before a rollout, the sim rollout
runner resolves an episode before it replays one, and the teleoperation tool
hands the resolved root to ``lerobot-record``. They lived in
:mod:`strands_robots.dataset_recorder`, in ``app``, so the rollout runner reached
UP a layer to find out where its own recording went.

Addressing a dataset is not writing one. These sit below the session that
writes, for the same reason
:mod:`strands_robots.dataset_metadata` sits below the surfaces that verify one: a
resolution that lives above one of its callers is a path that caller reaches up
for, or re-derives -- and a re-derived dataset directory is a read that misses
the write.

They earn ``core`` by their dependencies: the standard library,
:func:`~strands_robots._dyld.quiet_video_backend` and
:func:`~strands_robots.utils.non_negative_whole_number_error`, plus LeRobot
itself, imported inside the two functions that need it so a machine without it
still resolves a local path.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from strands_robots._dyld import quiet_video_backend
from strands_robots.utils import non_negative_whole_number_error


def _quiet_backend_kwargs(dataset_cls: Any) -> dict[str, Any]:
    """``{"video_backend": "pyav"}`` when torchcodec cannot load, else ``{}``.

    For the read-back constructors, which take no ``video_backend`` from the
    caller: the same one-line choice :func:`quiet_video_backend` makes for the
    recorder, guarded on the LeRobot version accepting the parameter.
    """
    if "video_backend" not in inspect.signature(dataset_cls).parameters:
        return {}
    resolved = quiet_video_backend()
    return {"video_backend": resolved} if resolved is not None else {}


def _lerobot_home() -> Path:
    """Return LeRobot's on-disk dataset home (``$HF_LEROBOT_HOME``).

    Uses lerobot's own ``HF_LEROBOT_HOME`` constant when importable so the
    resolved path matches exactly where ``LeRobotDataset`` reads/writes
    (honouring the ``HF_LEROBOT_HOME`` environment override). Falls back to the
    documented default ``~/.cache/huggingface/lerobot`` when lerobot is absent.
    """
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except (ImportError, ValueError, RuntimeError):
        return Path.home() / ".cache" / "huggingface" / "lerobot"


def local_dataset_dir(repo_id: str) -> Path | None:
    """The local directory a ``repo_id`` that is itself a path names.

    A ``repo_id`` that is absolute, ``./``-prefixed, or carries no
    ``owner/name`` slash is read as a local directory. That reading is this
    repo's, not LeRobot's - ``LeRobotDataset`` resolves any absent root to
    ``$HF_LEROBOT_HOME/{repo_id}`` whatever the id looks like - so these are
    exactly the ids whose directory this repo has to state on every surface that
    opens a dataset by id. Stating it on some of them and not others puts the
    directory written to and the directory read back in two different places.

    Args:
        repo_id: HuggingFace dataset id (``owner/name``) or a local path.

    Returns:
        The directory the id names, or None for an ``owner/name`` Hub id.

        ``None`` is a resolution, not a gap: that id's directory is LeRobot's to
        derive, and a *reader* must leave it there. An absent root is how
        LeRobot selects the revision-safe Hub snapshot cache for a download
        (``snapshot_download(cache_dir=HF_LEROBOT_HUB_CACHE)``); naming the
        directory instead switches it to a plain ``local_dir=`` materialization
        and skips the re-download a legacy on-disk layout triggers. A *writer*
        must never open that shared cache, which is why
        :func:`resolve_dataset_dir` names ``$HF_LEROBOT_HOME/{repo_id}`` for the
        same id - the two are the same directory whenever it already exists
        locally, and they differ only in who owns a download.
    """
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
        return Path(repo_id)
    return None


def resolve_dataset_dir(repo_id: str, root: str | None = None) -> Path:
    """Resolve the on-disk directory a dataset will be WRITTEN to.

    * explicit ``root`` -> used verbatim;
    * a ``repo_id`` that is itself a path -> the directory it names
      (:func:`local_dataset_dir`);
    * otherwise ``$HF_LEROBOT_HOME/{repo_id}``.

    Every writing entry point hands the result down as an explicit ``root``
    rather than letting LeRobot resolve a second time - otherwise the directory
    inspected before the write and the directory written to are two different
    places for exactly the ids the middle rule covers.

    This is the writer's resolution: the third rule names a concrete directory
    for a Hub id because a writer must not be handed LeRobot's shared snapshot
    cache. A reader resolves the middle rule only and leaves the third to
    LeRobot; see :func:`local_dataset_dir` on why the two differ.

    Args:
        repo_id: HuggingFace dataset id (``owner/name``) or a local path.
        root: Explicit local dataset directory, if any.

    Returns:
        The resolved dataset directory as a :class:`~pathlib.Path`.
    """
    if root:
        return Path(root)
    if (local := local_dataset_dir(repo_id)) is not None:
        return local
    return _lerobot_home() / repo_id


def load_lerobot_episode(repo_id: str, episode: int = 0, root: str | None = None) -> tuple[Any, int, int]:
    """Load a LeRobotDataset and resolve the frame range for an episode.

    Args:
        repo_id: HuggingFace dataset id.
        episode: Episode index, a non-negative whole number. Any real scalar
            with an integral value is accepted (a ``2.0`` from a config, a
            ``np.int64`` from arithmetic); the value is coerced with ``int()``
            once the shared guard has round-tripped it, so an accepted index
            reaches the O(1) episode-row lookup rather than the last-resort
            frame scan a float index falls through to.
        root: Local dataset directory. When omitted, a ``repo_id`` that is
            itself a path is read as the directory it names - the same one the
            recording entry points write to, so the id that recorded a dataset
            reads it back. An ``owner/name`` id keeps an absent root so LeRobot
            resolves its own revision-safe cache
            (:func:`~strands_robots.dataset_source.local_dataset_dir`).

    Returns:
        Tuple of (dataset, episode_start, episode_length) on success.

    Raises:
        ImportError: If lerobot is not installed.
        ValueError: If the episode index is not a usable non-negative whole
            number, is out of range, or the resolved episode has no frames.
    """
    # The domain is the shared non-negative whole-number rule, not a bare
    # ``< 0`` test. That test gave a verdict to three classes of value it
    # could not actually honor:
    #
    # * ``bool`` passed it (``True < 0`` is False) and then indexed the
    #   episode table as an int, so ``episode=True`` resolved **episode 1**
    #   and returned it as a success - a different episode than any caller
    #   passing a flag could have meant.
    # * A non-integral or non-finite value passed it too and was blamed on
    #   the dataset after a full-length boundary scan ("Episode 2.5 has no
    #   frames"), naming the data rather than the index.
    # * A str/list/None reached the comparison itself and raised
    #   ``TypeError``, which is not the ``ValueError`` this function
    #   documents as its refusal channel.
    #
    # Shared with the ``replay_episode`` teleop knob rather than restated:
    # that parameter is the same quantity on a neighbouring surface, and
    # ``non_negative_whole_number_error`` already names it.
    if msg := non_negative_whole_number_error(episode, "episode", "load_lerobot_episode"):
        raise ValueError(msg)
    # Safe because the guard performed this coercion and compared the result
    # back; see its docstring on why the two steps are ordered this way.
    episode = int(episode)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # The directory to read, resolved by the same rule the recording was written
    # through. A ``repo_id`` that is itself a path is a local directory here as
    # it is there (:func:`local_dataset_dir`); forwarding the caller's ``None``
    # unresolved sent the read somewhere the recording never was, because
    # LeRobot reads an absent root as ``$HF_LEROBOT_HOME/{repo_id}`` whatever
    # the id looks like. So the id that recorded a dataset could not read it
    # back: the miss falls through to a Hub lookup for a dataset name that only
    # ever named a directory.
    #
    # Only that rule is resolved here. An ``owner/name`` id keeps its absent
    # root, which is how LeRobot selects the revision-safe snapshot cache for a
    # download - and it already reads back what a local write put at
    # ``$HF_LEROBOT_HOME/{repo_id}``, since that is the same directory LeRobot
    # derives. Resolving it here would move Hub downloads out of that cache for
    # no gain.
    read_root = Path(root) if root else local_dataset_dir(repo_id)
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(read_root) if read_root is not None else None,
        **_quiet_backend_kwargs(LeRobotDataset),
    )

    num_episodes = ds.meta.total_episodes if hasattr(ds.meta, "total_episodes") else len(ds.meta.episodes)
    if episode >= num_episodes:
        raise ValueError(f"Episode {episode} out of range (0-{num_episodes - 1})")

    episode_start = 0
    episode_length = 0
    try:
        ep_info = ds.meta.episodes[episode] if hasattr(ds.meta, "episodes") else {}
        if "dataset_from_index" in ep_info:
            # LeRobot 0.6 records the range on the episode's own metadata row,
            # so this is one row read whatever the index is. Every lerobot in
            # the declared range writes those columns, which is why this rung
            # leads: the two below it are compatibility fallbacks, and the
            # ``length`` accumulation reads one row per *preceding* episode to
            # recompute a number this row already states.
            episode_start = int(ep_info["dataset_from_index"])
            episode_length = int(ep_info["dataset_to_index"]) - episode_start
        elif hasattr(ds, "episode_data_index"):
            from_idx = ds.episode_data_index["from"][episode].item()
            to_idx = ds.episode_data_index["to"][episode].item()
            episode_start = from_idx
            episode_length = to_idx - from_idx
        else:
            for i in range(episode):
                prior_info = ds.meta.episodes[i] if hasattr(ds.meta, "episodes") else {}
                episode_start += prior_info.get("length", 0)
            episode_length = ep_info.get("length", 0)
    except Exception:
        # Last resort: scan frames to find episode boundaries
        for idx in range(len(ds)):
            frame = ds[idx]
            frame_ep = frame.get("episode_index", -1) if hasattr(frame, "get") else -1
            if hasattr(frame_ep, "item"):
                frame_ep = frame_ep.item()
            if frame_ep == episode:
                if episode_length == 0:
                    episode_start = idx
                episode_length += 1
            elif episode_length > 0:
                break

    if episode_length == 0:
        raise ValueError(f"Episode {episode} has no frames")

    return ds, episode_start, episode_length
