---
description: The four tools a VLM judge drives to read a recorded episode - frame sampling with per-camera image blocks, the authoritative predicate verdict, the annotation write - and how to measure judge/human agreement against its baseline before filtering training data.
---

# The episode judge (tools and calibration)

The judge annotates on top of the deterministic verdict and can never overturn
it; [Episode labels](episode-labels.md) describes that two-stage verdict and the
sidecar the annotations land in. This page is the agent that writes them and the
measurement that says whether to trust it.

## The four tools

Four tools drive the labeling, assembled by
`strands_robots.tools.episode_judge.create_judge_agent` - model-provider
agnostic (any strands model object; a local OpenAI-compatible VLM endpoint
works, no cloud dependency required):

- `load_episode` - frame count, features, whether a verdict/label exists.
- `sample_frames` - evenly spaced state vectors plus a motion summary:
  `rms_state_jerk` (rms third difference of the state series - jerk is the
  third derivative, so this is the field that grounds `jerky_motion` from
  state alone) and `max_state_delta` (peak per-step delta, for spotting
  discontinuities); `include_images=True` decodes the camera frames into
  image blocks for a multimodal judge (needs the `lerobot` extra). Every
  recorded camera is included - one block per camera per sampled position,
  position-major, cameras in sorted order, with the count and grouping
  stated in the leading text block and every image block immediately
  preceded by a text block naming its camera and `frame_index`, so a
  per-view observation can be attributed to a view and joined back onto that
  frame's state row. The label sits *before* its image because a text
  block is read as a caption of the image that follows it. No view is
  canonical: the same motion can sit above a judge's legibility threshold in
  one camera and below it in another, so sampling one of them would drop
  verdicts the interleave keeps.
  The state is always reported as a vector, however narrow: LeRobot stores a
  one-component state as a scalar column rather than a one-element list, and
  both are read as the vector `meta/info.json` declares, so a single-DOF
  recording (a gripper, a linear stage, a pan unit) samples like any other.
  An episode whose frames are not all readable is refused rather than
  summarised: both statistics are computed over consecutive frames, so a data
  shard lost to a truncated download would make the surviving rows read as
  consecutive across a gap and the summary would measure that gap instead of
  the robot. The refusal names every unreadable shard. `load_episode` reads the
  episode metadata rather than the frames, so it still describes such an
  episode - including its true length.
- `read_predicate_verdict` - the authoritative deterministic verdict.
- `write_label` - the annotation; structurally unable to touch the verdict.

All four answer with the `{"status", "content"}` envelope even when the
parquet reader is missing. `pyarrow` ships with the `lerobot` extra, so a judge
process that only reads datasets recorded elsewhere can be running without it;
`load_episode` and `sample_frames` then refuse by naming the extra to install,
and the two sidecar tools (`read_predicate_verdict`, `write_label`) read JSON
and are unaffected. A judge run over a hundred episodes reports the episode it
could not read rather than dying on it, so no tool raises past the dispatch.

Two failure modes lean on judge capability rather than on a payload field,
so calibrate before trusting them: `jerky_motion` is grounded for a
text-only judge by `rms_state_jerk`, and `camera_occlusion` is a claim about
*one* view, so it needs a payload in which that view can be named - each
image block carries its camera, which is what makes the tag expressible at
all. Naming the view is not the same as emitting the tag over a real dataset,
so still calibrate (`measure_agreement`) before trusting `camera_occlusion`,
and treat any direction phrase in a free-text `note` as a statement about a
camera frame, not about the world.

The `note` is for humans and is never parsed by anything downstream: the
filterable channels are the closed vocabularies, and a free-text description
of a still frame is the least reliable thing a judge emits.

```python
from strands.models import BedrockModel  # or any strands model provider
from strands_robots.tools.episode_judge import create_judge_agent

judge = create_judge_agent(model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-v1:0"))
judge("Label every episode of the dataset at /data/pick_place. "
      "Use sample_frames with include_images=True to look at the recording.")
```

## Calibration before trust

Measure judge/human agreement on a small human-labeled holdout before letting
the judge filter training data - the measurement ships with the pipeline
(`measure_agreement`), not as a promise:

```python
from strands_robots.episode_labels import measure_agreement

report = measure_agreement("/data/pick_place", {
    3: {"quality": "high", "failure_mode": None},
    7: {"quality": "low", "failure_mode": "jerky_motion"},
})
print(report["quality_agreement"], report["quality_baseline"], report["disagreements"])
```

Read each agreement fraction against the baseline reported beside it, never on
its own. Both fractions are accuracies over a column with a class balance, and
a recorded dataset is mostly clean, so a judge that emitted one label for every
episode already scores the majority-class frequency having read nothing.
`quality_baseline` and `failure_mode_baseline` are what that constant answer
earns on the same holdout, over exactly the episodes compared, so a fraction at
or below its baseline says the judge is indistinguishable from one that read
nothing - however high the fraction reads. A judge is calibrated by the gap, not by the
fraction. Both baselines are `None` in step with the fraction they accompany.

Read the gap per tag rather than once, because the taxonomy is not uniformly
legible: four evenly spaced stills cannot show jitter, so a `jerky_motion` tag
can sit at its base rate on the same payload that separates
`camera_occlusion` cleanly. Filter on the tags whose gap is real.

## See also

- [Episode labels](episode-labels.md) - the sidecar schema, field domains and the
  filtered re-training step.
- [Verify a dataset](verifying-datasets.md) - the deterministic checks the judge
  layers on top of.
