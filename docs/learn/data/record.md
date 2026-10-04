# Record

At the end of this page a control loop, in simulation or on hardware, writes a LeRobot v3 dataset (parquet plus one MP4 per camera per episode) that `lerobot-train` and every policy provider here can read, and you know where it lands on disk and how to keep episodes distinct.

Recording needs the `[lerobot]` extra for the dataset schema. On the simulation side one call opens a session and `run_policy` writes one frame per control step while it is open:

```python title="sketch"
from strands_robots import Robot

sim = Robot("so101")
sim.add_camera("front", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1])
sim.start_recording(repo_id="you/so101_reach", task="reach the cube", fps=30, cameras=["front"])
sim.run_policy(policy_provider="mock", instruction="reach the cube", duration=3.0, n_episodes=5, reset_between=True)
print(sim.stop_recording())              # flushes the last episode to parquet, finalizes meta/
print(sim.verify_dataset_episodes(expected=5))
```

The mock does not read the instruction (`reads_instruction = False`), and every task report says so: `Note: MockPolicy does not read the instruction`. The five episodes are sinusoid test motions labelled "reach the cube"; the task label describes the intent, not the motion. Swap in a real provider or a teleoperator before training on what you record.

## Where it goes

`repo_id` is a Hub-shaped name, `org/name`. `root=` overrides the directory; otherwise the dataset lands under `$HF_LEROBOT_HOME/<repo_id>` (default `~/.cache/huggingface/lerobot/<repo_id>`). `resolve_dataset_dir(repo_id, root)` is the one function every layer uses to answer "which directory", so the recorder, the rollout runner and `replay_episode` agree.

Layout after `stop_recording()`:

```text
<root>/
  meta/info.json            fps, features, total_episodes, total_frames, video_path template
  meta/episodes/**.parquet  the ground truth: one row per episode, episode_index + length
  data/**.parquet           one row per frame: observation.state, action, timestamp, task
  videos/<camera>/**.mp4    observation.images.<camera>, several episodes packed per file
  episode_labels.json       optional, written by the judge (see label-and-judge.md)
```

## Sim session verbs

| verb | does |
|---|---|
| `start_recording(repo_id, task, fps=30, root=None, vcodec="h264", overwrite=False, cameras=None)` | declares the schema from the live model: every joint of every robot, every named camera |
| `run_policy(..., n_episodes=, reset_between=True)` | records one `(observation, action)` frame per control step with the observation re-sampled at that step |
| `save_episode()` | closes the open episode by hand when you step the loop yourself |
| `stop_recording(push_to_hub=False, bucket=None, run_id=None)` | flushes, finalizes, optionally pushes or syncs ([stream and sync](stream-and-sync.md)) |
| `get_recording_status()` | `recording`, `steps`, `last_save` |
| `replay_episode(repo_id, episode=0, speed=1.0)` | plays a recorded action stream back into the world |
| `start_cameras_recording()` | plain MP4 with no dataset schema; needs `[sim-mujoco]` or `[sim-isaac]` (Isaac matches the MuJoCo filename convention so cross-backend tooling finds both the same way) |

`overwrite=True` deletes the existing directory, and with it the label sidecar. A second `start_recording` on an open session is refused; stop the first.

## The recorder itself

`DatasetRecorder` is the class under both the simulation and any hardware script. One `add_frame` per step:

```python title="sketch"
from strands_robots.dataset_recorder import DatasetRecorder

rec = DatasetRecorder.create(
    repo_id="you/so101_real", fps=30, robot_type="so101",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    camera_keys=["front"], camera_dims={"front": (480, 640)}, task="pick the cube",
)
for observation, action in loop():            # your control loop
    rec.add_frame(observation, action, task="pick the cube")
rec.save_episode()                             # once per episode
rec.finalize()
```

`add_frame` refuses a frame missing a declared column (`ValueError`), reports a lost frame (`RecordingFrameError`), drops undeclared action keys; `create` normalises `vcodec` (`libx264` becomes `h264`). `DatasetRecorder.resume(repo_id)` appends to an existing dataset with the same schema.

## On hardware

`lerobot_teleoperate(action="start", robot_type="so101_follower", robot_port=..., teleop_type="so101_leader", teleop_port=..., dataset_repo_id="you/so101_real", dataset_single_task="pick the cube", dataset_num_episodes=50, dataset_fps=30, dataset_episode_time_s=60, dataset_reset_time_s=60)` runs lerobot's own `lerobot-record` as a managed session with cameras from `robot_cameras=`. The result is the same layout, so the same [verify](verify.md) and [label](label-and-judge.md) steps apply.

## The one failure to know

To watch the arm and its cameras while a recording runs, or to open a finished episode in Foxglove, see [Foxglove](foxglove.md).

A run that intended N episodes but never called `save_episode` between them writes one `episode_index=0` mega-episode with the right total frame count. Nothing in the recorder's bookkeeping catches that; the parquet does. Run `strands-robots verify-dataset <root> --expected N` before training, every time.
