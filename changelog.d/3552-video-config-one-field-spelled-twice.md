### Fixed: a `video` field spelled twice is refused, not resolved to one of the two

`run_policy` / `eval_policy` / `evaluate_benchmark` take recording options as a
free-form `video={...}` dict, and each field there has more than one accepted
spelling (`path` / `output_path` / `record_video`, `camera` / `camera_name` /
`video_camera`, `fps` / `video_fps`, and the same for `width` and `height`).
The resolver took the first spelling that carried a value and dropped the rest,
so a dict naming two of them honored one in silence -
`{"path": p, "camera": "top", "camera_name": "wrist"}` recorded the whole
rollout from `top` under `status="success"` while the caller had also named
`wrist`, and `{"path": a, "output_path": b}` wrote `a` and left `b` absent.
The canonical spelling won regardless of order, so a caller-supplied legacy
spelling lost to a wrapper's default.

That is the drop this schema already refuses for an unknown key, reached
through keys it accepts. Two spellings of one field that disagree are now a
structured error naming both keys and both values; two carrying the same value
discard nothing and are still honored, as is a single spelling of any field.
