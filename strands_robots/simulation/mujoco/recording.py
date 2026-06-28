"""Recording mixin - start/stop trajectory recording to LeRobotDataset."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class RecordingMixin:
    """Trajectory recording mixed into ``Simulation``.

    Writes per-step observations + actions + instruction to a LeRobotDataset
    via ``start_recording`` / ``stop_recording`` and the ``on_frame`` hook
    in ``PolicyRunner``. Separately from that, ``start_cameras_recording``
    dumps raw per-camera MP4s.

    **Coupling** (see simulation.py top-level docstring): mixin reaches
    into ``self._world`` (trajectory buffer + dataset_recorder live in
    ``_world._backend_state``). ``TYPE_CHECKING`` stub below exists so mypy
    accepts the ``_world`` lookup; it is a documentary contract, not an
    enforceable protocol.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: "SimWorld | None"
        default_width: int
        default_height: int

    def _is_recording(self) -> bool:
        """True when a dataset-recording session is active.

        Overrides :meth:`SimEngine._is_recording` so the multi-episode
        ``run_policy`` loop flushes an episode boundary after each rollout
        only while a recording is open.
        """
        return self._world is not None and bool(self._world._backend_state.get("recording", False))

    def _active_recorder(self) -> Any:
        """Live dataset recorder, or ``None`` when no session is open.

        Overrides :meth:`SimEngine._active_recorder` so the base ``run_policy``
        episode-contract fields can read the recorder's in-memory episode count.
        """
        if self._world is None:
            return None
        return self._world._backend_state.get("dataset_recorder")

    def _active_dataset_root(self) -> str | None:
        """On-disk root of the active or most-recently-recorded dataset.

        Overrides :meth:`SimEngine._active_dataset_root` so
        :meth:`verify_dataset_episodes` can locate the parquet AFTER
        ``stop_recording`` has finalized it and dropped the recorder. Prefers
        the live recorder's root; falls back to the ``last_dataset_root`` stashed
        at ``start_recording``.
        """
        recorder = self._active_recorder()
        if recorder is not None:
            try:
                return str(recorder.root)
            except (AttributeError, TypeError):
                pass
        if self._world is None:
            return None
        last = self._world._backend_state.get("last_dataset_root")
        return str(last) if last else None

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Start recording to LeRobotDataset format (parquet + per-camera MP4).

        Requires the ``lerobot`` extra for the dataset schema. If you only
        need plain MP4 video (no dataset schema, no policy-training metadata),
        use :meth:`start_cameras_recording` - it runs under the
        ``[sim-mujoco]`` extra alone (imageio-ffmpeg backend).

        Raises:
            Friendly error when ``lerobot`` is not installed, directing the
            caller to :meth:`start_cameras_recording` or to install the
            optional extra.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        _DatasetRecorder: Any = None
        _has_lerobot = False
        try:
            from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
            from strands_robots.dataset_recorder import has_lerobot_dataset as _check_lerobot

            _has_lerobot = _check_lerobot()
        except ImportError:
            pass

        if not _has_lerobot or _DatasetRecorder is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "start_recording produces a LeRobotDataset (parquet + video) and "
                            "requires the lerobot extra. For plain MP4 video under the "
                            "[sim-mujoco] extra alone, use start_cameras_recording instead.\n"
                            "\n"
                            "  - Dataset + policy training data:  pip install 'strands-robots[lerobot]'\n"
                            "  - Plain MP4 only:                  start_cameras_recording(cameras=..., output_dir=...)"
                        )
                    }
                ],
            }

        self._world._backend_state["recording"] = True
        self._world._backend_state["trajectory"] = []
        self._world._backend_state["push_to_hub"] = push_to_hub

        # Resolve the on-disk dataset dir (shared by overwrite + resume logic).
        if root:
            dataset_dir = Path(root)
        elif "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
            dataset_dir = Path(repo_id)
        else:
            dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        # Stash the resolved root so verify_dataset_episodes can read the parquet
        # after stop_recording has finalized the dataset and dropped the recorder.
        self._world._backend_state["last_dataset_root"] = str(dataset_dir)

        # Multi-episode append: when NOT overwriting and a dataset already
        # exists on disk, resume it (append new episodes) instead of calling
        # create() - which hard-fails with FileExistsError (B12). resume() is
        # the only correct append path in LeRobot 0.5.2+ (the plain constructor
        # is read-only). When overwrite=True, wipe and recreate from scratch.
        resume_existing = (
            not overwrite and dataset_dir.exists() and dataset_dir.is_dir() and (dataset_dir / "meta").exists()
        )

        try:
            if overwrite:
                if dataset_dir.exists() and dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir)
                    logger.info("Removed existing dataset dir: %s", dataset_dir)

            # Collect joint names from every robot. When the scene contains
            # more than one robot (e.g. multi-agent dual-task recording), prefix
            # each joint with the robot's instance name (``alice__shoulder_pan``)
            # so the dataset schema has unique joint ids per agent. Single-robot
            # scenes keep the clean ``shoulder_pan`` names for backwards compat.
            joint_names: list[str] = []
            camera_keys: list[str] = []
            robot_type = "unknown"
            multi_robot = len(self._world.robots) > 1
            for rname, robot in self._world.robots.items():
                if multi_robot:
                    joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                else:
                    joint_names.extend(robot.joint_names)
                robot_type = robot.data_config or rname

            mj = _ensure_mujoco()
            # Declare each camera in the dataset schema at the SAME
            # resolution it actually renders at. Cameras added via add_camera
            # carry their own width/height (e.g. 256x256 for a LIBERO VLA),
            # which can differ from the sim's default render size. Declaring
            # everything at default_width/height made add_frame reject frames
            # ("shape (256,256,3) != expected (3,480,640)") and, with strict
            # recording, abort the whole episode. We map each safe camera name
            # to its real (height, width) so _build_features sizes it correctly.
            camera_dims: dict[str, tuple[int, int]] = {}
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name:
                    continue
                # LeRobot feature names can't contain '/' (reserved for
                # nested-feature addressing). When a robot injects a
                # namespaced camera (e.g. ``arm0/wrist_cam``), collapse
                # the separator to ``__`` for the dataset schema.
                safe_name = cam_name.replace("/", "__")
                camera_keys.append(safe_name)
                cam_info = self._world.cameras.get(cam_name) or self._world.cameras.get(safe_name)
                if cam_info is not None:
                    camera_dims[safe_name] = (int(cam_info.height), int(cam_info.width))
                else:
                    camera_dims[safe_name] = (int(self.default_height), int(self.default_width))

            assert _DatasetRecorder is not None  # checked above
            if resume_existing:
                # Append to the existing dataset (schema inherited from disk).
                logger.info("Resuming existing dataset for append: %s", dataset_dir)
                resumed = _DatasetRecorder.resume(
                    repo_id=repo_id,
                    root=root,
                    task=task,
                    vcodec=vcodec,
                )
                # resume() inherits the feature schema from disk; it does NOT
                # check it against the CURRENT scene. Adding a robot or swapping
                # a camera resolution between episodes would otherwise yield a
                # cryptic per-feature shape error on the next add_frame. Compare
                # up front and raise a clear schema-diff instead.
                self._verify_resume_schema(resumed, joint_names, camera_keys, camera_dims)
                self._world._backend_state["dataset_recorder"] = resumed
            else:
                self._world._backend_state["dataset_recorder"] = _DatasetRecorder.create(
                    repo_id=repo_id,
                    fps=fps,
                    robot_type=robot_type,
                    joint_names=joint_names,
                    camera_keys=camera_keys,
                    camera_dims=camera_dims,
                    task=task,
                    root=root,
                    vcodec=vcodec,
                    video_width=self.default_width,
                    video_height=self.default_height,
                )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Recording to LeRobotDataset: {repo_id}\n"
                            f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                            f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                            f"Run policies to capture frames, then stop_recording to save episode"
                        )
                    }
                ],
            }
        except Exception as e:
            self._world._backend_state["recording"] = False
            logger.error("Dataset recorder init failed: %s", e)
            return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}

    def _verify_resume_schema(
        self,
        recorder: Any,
        joint_names: list[str],
        camera_keys: list[str],
        camera_dims: dict[str, tuple[int, int]],
    ) -> None:
        """Verify the live scene matches the resumed dataset's on-disk schema.

        ``DatasetRecorder.resume`` inherits the feature schema from disk; it does
        not validate it against the current scene. If the caller added a robot,
        renamed a joint, or changed a camera resolution between episodes, the
        mismatch would only surface as a cryptic per-feature shape error on the
        next ``add_frame``. Compare here and raise a clear schema diff instead.

        Compares the expected ``observation.state`` joint names and each
        ``observation.images.*`` camera (presence + height/width). Best-effort:
        if the dataset does not expose ``features`` we skip silently rather than
        block a valid resume on an unexpected LeRobot layout.

        Args:
            recorder: The resumed DatasetRecorder.
            joint_names: Joint names the current scene will emit (namespaced for
                multi-robot scenes).
            camera_keys: Sanitized camera feature names the current scene emits.
            camera_dims: Map of camera feature name -> (height, width).

        Raises:
            ValueError: If the live scene schema diverges from the on-disk one.
        """
        features = getattr(getattr(recorder, "dataset", None), "features", None)
        if not isinstance(features, dict):
            return

        diffs: list[str] = []

        state = features.get("observation.state")
        if isinstance(state, dict):
            disk_joints = list(state.get("names") or [])
            if disk_joints and disk_joints != list(joint_names):
                diffs.append(f"observation.state joints differ: on-disk={disk_joints} vs scene={list(joint_names)}")

        for cam in camera_keys:
            key = f"observation.images.{cam}"
            disk_cam = features.get(key)
            if not isinstance(disk_cam, dict):
                diffs.append(f"camera '{cam}' is in the scene but not in the on-disk schema")
                continue
            shape = disk_cam.get("shape")
            scene_dim = camera_dims.get(cam)
            if shape and len(shape) == 3 and scene_dim is not None:
                _, disk_h, disk_w = shape
                scene_h, scene_w = scene_dim
                if (int(disk_h), int(disk_w)) != (int(scene_h), int(scene_w)):
                    diffs.append(
                        f"camera '{cam}' resolution differs: on-disk={(disk_h, disk_w)} vs scene={(scene_h, scene_w)}"
                    )

        disk_cams = {k[len("observation.images.") :] for k in features if k.startswith("observation.images.")}
        for cam in disk_cams - set(camera_keys):
            diffs.append(f"camera '{cam}' is in the on-disk schema but not in the current scene")

        if diffs:
            raise ValueError(
                "Cannot resume recording: the current scene does not match the existing dataset schema. "
                "Use overwrite=True for a fresh dataset, or restore the original scene. Differences:\n  - "
                + "\n  - ".join(diffs)
            )

    def stop_recording(
        self,
        output_path: str | None = None,
        *,
        push_to_hub: bool = False,
        bucket: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Stop recording and save episode to LeRobotDataset.

        idempotent - calling when not recording succeeds with a
        'Was not recording' message so callers can safely call it unconditionally.

        Returns a structured ``status="error"`` when the recording captured no
        frames (the dataset would contain only ``meta/info.json``), rather than
        silently writing an empty dataset. Only ``run_policy`` feeds the active
        recorder (via its ``on_frame`` hook); ``eval_policy`` / ``evaluate`` /
        ``replay_episode`` and bare ``step`` loops do not, so recording around
        those produces zero frames and is reported as an error.

        Args:
            output_path: Unused legacy arg (kept for back-compat).
            push_to_hub: Publish to a versioned HF *dataset* repo (the finished
                artifact). Overrides the ``push_to_hub`` set at start_recording.
            bucket: If set (e.g. ``"my-org/robot-fave"``), sync the dataset into
                a mutable HF Storage Bucket instead of/in addition to the dataset
                repo - the Phase 1/2 collection target (Xet-deduped, overwrite in
                place). See reports/STREAMING_DATA_LOOP_DEEP_DIVE.md §2.4 / App. A.
            run_id: Optional subpath inside the bucket (defaults to dataset name).
        """
        if self._world is None or not self._world._backend_state.get("recording", False):
            return {"status": "success", "content": [{"text": "Was not recording."}]}

        self._world._backend_state["recording"] = False
        recorder = self._world._backend_state.get("dataset_recorder", None)

        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        # Save the trailing (unsaved) episode, then guard against an empty
        # dataset. ``episode_frame_count`` is the frames captured since the last
        # save_episode; ``frame_count`` is the monotonic total across the
        # dataset. Three cases:
        #
        #   1. Unsaved frames pending (episode_frame_count > 0): flush them with
        #      save_episode. If LeRobot rejects the flush, surface the error.
        #   2. No pending frames but the dataset already has some (callers that
        #      save per-episode and call stop_recording last): nothing to flush,
        #      just finalize - calling save_episode here would hit LeRobot's
        #      "add frames before add_episode" guard on the empty buffer and
        #      wrongly fail an otherwise-complete dataset.
        #   3. Nothing ever captured (frame_count == 0): fail loudly instead of
        #      writing a 0-frame dataset. This happens when the rollout was
        #      driven by eval_policy / evaluate / replay_episode or a bare step
        #      loop - none of which feed the active recorder (only run_policy's
        #      on_frame hook calls add_frame). Previously stop_recording reported
        #      success with "0 frames, 0 episode(s)", silently producing a
        #      dataset with only meta/info.json (no parquet/video).
        pending = getattr(recorder, "episode_frame_count", 0)
        captured = getattr(recorder, "frame_count", 0)
        if pending > 0:
            save_result = recorder.save_episode()
            if isinstance(save_result, dict) and save_result.get("status") == "error":
                self._world._backend_state["dataset_recorder"] = None
                self._world._backend_state["trajectory"] = []
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "stop_recording failed to save the final episode "
                                f"({pending} pending frames). save_episode: "
                                f"{save_result.get('message')}"
                            )
                        }
                    ],
                }
        elif captured == 0:
            self._world._backend_state["dataset_recorder"] = None
            self._world._backend_state["trajectory"] = []
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "stop_recording captured no frames - dataset would be empty "
                            "(0 frames). Frames are written only by run_policy(...) while "
                            "recording is active (its on_frame hook calls add_frame). "
                            "eval_policy / evaluate / replay_episode and bare step loops do "
                            "NOT feed the recorder. To record a dataset: start_recording -> "
                            "run_policy (once per episode) -> stop_recording."
                        )
                    }
                ],
            }

        repo_id = recorder.repo_id
        frame_count = recorder.frame_count
        episode_count = recorder.episode_count
        root = recorder.root

        # Finalize FIRST so meta/ (stats/info) is written before any bucket sync
        # - streaming/training downstream needs it (App. F.2).
        recorder.finalize()

        # #708 - parquet-truth gate. The recorder's ``episode_count`` is the
        # author-side bookkeeping (incremented by every ``save_episode`` call).
        # The dataset's own ``meta.total_episodes`` / parquet rowcount is what
        # downstream consumers (HF hub, training loaders, audit tools) trust.
        # If they disagree, the on-disk dataset is the source of truth (Law-7
        # in AGENTS.md: parquet num_rows > meta/info.json > markdown). Surface
        # the mismatch in the returned payload so the caller - and any CI that
        # parses the status dict - can fail loudly instead of shipping a
        # silent-collapse dataset.
        parquet_episode_count: int | None = None
        episode_count_mismatch: bool = False
        episode_count_mismatch_orig: int = episode_count
        try:
            ds_meta = getattr(getattr(recorder, "dataset", None), "meta", None)
            if ds_meta is not None:
                parquet_episode_count = int(getattr(ds_meta, "total_episodes", 0))
                if parquet_episode_count != episode_count:
                    episode_count_mismatch = True
                    logger.warning(
                        "stop_recording: recorder.episode_count=%d but "
                        "dataset.meta.total_episodes=%d. Trust the parquet. "
                        "(#708 silent-collapse gate)",
                        episode_count,
                        parquet_episode_count,
                    )
                    # The parquet is the ground truth - report it as the
                    # canonical episode_count downstream. Stash the original
                    # recorder.episode_count so the text payload can name both.
                    episode_count_mismatch_orig = episode_count
                    episode_count = parquet_episode_count
        except Exception as e:  # noqa: BLE001 - never fail finalize on a probe
            logger.debug("episode_count gate probe failed: %s", e)

        extra = ""
        # Bucket sync (Phase 1/2): mutable, Xet-deduped collection dump.
        if bucket:
            sync_result = recorder.sync_to_bucket(bucket, run_id=run_id)
            if sync_result.get("status") == "success":
                extra += f"\nSynced to bucket: {sync_result['bucket_uri']}"
            else:
                extra += f"\nBucket sync FAILED: {sync_result.get('message')}"
        # Versioned dataset-repo publish (Phase 4 hand-off).
        if push_to_hub or self._world._backend_state.get("push_to_hub", False):
            push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])
            if push_result and push_result.get("status") == "success":
                extra += "\nPushed to HuggingFace Hub"
            elif push_result:
                extra += f"\npush_to_hub FAILED: {push_result.get('message')}"

        self._world._backend_state["dataset_recorder"] = None
        self._world._backend_state["trajectory"] = []

        # #708 - if recorder.episode_count and parquet disagree, surface
        # it in the human-readable text too so an operator scanning the
        # status log sees the gate firing.
        if episode_count_mismatch:
            text_episode_note = (
                f"\n[#708 gate] recorder reported {episode_count_mismatch_orig} "
                f"episodes but parquet has {episode_count}. Trusting parquet."
            )
        else:
            text_episode_note = ""

        text = (
            f"Episode saved to LeRobotDataset\n"
            f"{repo_id} -- {frame_count} frames, {episode_count} episode(s)"
            f"{text_episode_note}\n"
            f"Local: {root}{extra}"
        )

        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": {
                        "repo_id": repo_id,
                        "frame_count": frame_count,
                        "episode_count": episode_count,
                        "parquet_episode_count": parquet_episode_count,
                        "episode_count_mismatch": episode_count_mismatch,
                        "root": root,
                    }
                },
            ],
        }

    def save_episode(self) -> dict[str, Any]:
        """Close the current episode and start a fresh one in the same session.

        This is the explicit episode-boundary primitive for multi-episode
        recording. The documented collection workflow is one ``run_policy``
        call per episode:

            sim.start_recording(repo_id=..., task=...)
            for _ in range(n_episodes):
                sim.run_policy(robot_name=..., n_steps=...)
                sim.save_episode()   # flush this rollout as its own episode
            sim.stop_recording()

        Without this call, every ``run_policy`` rollout in a session appends to
        the SAME buffer, so ``stop_recording`` flushes them as a single
        ``episode_index=0`` (1200 steps land in one episode instead of N). Each
        ``save_episode`` writes the buffered frames as a distinct episode with
        its own ``episode_index`` / ``length`` / ``from_index`` / ``to_index``
        and resets the per-episode frame buffer; ``stop_recording`` flushes any
        trailing rollout automatically, so a final ``save_episode`` is optional.

        Per-episode stats (LeRobot computes ``stats.json`` per episode, then
        aggregates) stay correct because each rollout's frames are isolated to
        their own episode rather than being mixed across mid-session
        ``reset()`` teleports.

        Idempotent on an empty buffer: when no frames have been captured since
        the last boundary (or since ``start_recording``), it succeeds with a
        "no frames to flush" message rather than tripping LeRobot's
        "add frames before add_episode" guard, so callers can invoke it
        unconditionally inside a loop.

        Returns:
            Standard status dict. On success the ``content`` text reports the
            episode index and frame count; a structured ``status="error"`` is
            returned when no recording is active or the underlying flush fails.
        """
        if self._world is None or not self._world._backend_state.get("recording", False):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "save_episode: not recording. Call start_recording first, "
                            "then run_policy (once per episode) -> save_episode -> stop_recording."
                        )
                    }
                ],
            }

        recorder = self._world._backend_state.get("dataset_recorder", None)
        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        pending = getattr(recorder, "episode_frame_count", 0)
        if pending <= 0:
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "save_episode: no frames to flush (buffer empty). Run a policy "
                            "while recording before closing an episode."
                        )
                    }
                ],
            }

        save_result = recorder.save_episode()
        if isinstance(save_result, dict) and save_result.get("status") == "error":
            # The recorder marks itself closed on a failed flush (the LeRobot
            # episode buffer is in an undefined state); drop it so callers do
            # not keep appending into a poisoned recorder.
            self._world._backend_state["recording"] = False
            self._world._backend_state["dataset_recorder"] = None
            self._world._backend_state["trajectory"] = []
            return {
                "status": "error",
                "content": [{"text": f"save_episode failed: {save_result.get('message')}"}],
            }

        # Reset the in-memory trajectory mirror so get_recording_status reports
        # the NEXT episode from zero (matching the recorder's per-episode reset).
        self._world._backend_state["trajectory"] = []

        episode = save_result.get("episode")
        ep_frames = save_result.get("episode_frames")
        total = save_result.get("total_frames")
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Episode {episode} saved -- {ep_frames} frames "
                        f"({total} total across dataset). Buffer reset for the next episode."
                    )
                }
            ],
        }

    def stream_dataset(self, repo_id: str, **kwargs):
        """Open a streaming reader for a LeRobotDataset - read frames straight
        from the Hub (or a local root) with no full materialization.

        This is the in-process counterpart to ``start_recording`` /
        ``stop_recording``: where those WRITE a dataset, ``stream_dataset``
        READS one back lazily for eval / replay / inspection (Phase 3 of the
        physical-AI data loop). Training scripts can instead use
        ``python -m lerobot.scripts.train dataset.streaming=true`` which uses
        the same underlying StreamingLeRobotDataset (deep-dive Appendix D).

        Args:
            repo_id: HF dataset id (e.g. ``"lerobot/svla_so100_pickplace"``) or
                a local repo_id paired with ``root=``.
            **kwargs: Forwarded to
                :meth:`StreamingDatasetReader.open` - e.g. ``root``,
                ``delta_timestamps``, ``episodes``, ``shuffle``, ``buffer_size``,
                ``max_num_shards``, ``drop_videos`` (proprio-only, torchcodec-free).

        Returns:
            A :class:`~strands_robots.streaming_dataset.StreamingDatasetReader`.

        Example:
            reader = sim.stream_dataset(
                "local/agent_demo", root="/tmp/strands_agent_dataset",
                delta_timestamps={"observation.state": [-0.0667, 0.0],
                                  "action": [0.0, 0.0667]},
                shuffle=False,
            )
            for frame in reader:
                ...
        """
        from strands_robots.streaming_dataset import StreamingDatasetReader

        return StreamingDatasetReader.open(repo_id, **kwargs)

    def get_recording_status(self) -> dict[str, Any]:
        """Returns success in every lifecycle state (no world / not
        recording / recording) with a distinguishing message so callers can
        poll it unconditionally without try/except."""
        if self._world is None:
            return {
                "status": "success",
                "content": [{"text": "No world. Call create_world to start recording."}],
            }

        recording = self._world._backend_state.get("recording", False)
        steps = len(self._world._backend_state.get("trajectory", []))

        if recording:
            text = f"[recording] {steps} steps captured"
        else:
            text = f"[idle] Not recording (last episode: {steps} steps)"

        return {
            "status": "success",
            "content": [{"text": text}],
        }
