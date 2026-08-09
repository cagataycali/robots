### Fixed: a `start_recording` posture flag is checked instead of read by truthiness

`start_recording(push_to_hub=, overwrite=)` and `stop_recording(push_to_hub=)` read two
*posture* flags by truthiness, so every non-empty string selected the branch the caller was
opting out of. `overwrite="false"` reached the wipe branch and deleted the dataset it was
meant to append to - measured on MuJoCo, a dataset holding one recorded episode came back
with that episode gone while `start_recording` returned `status="success"` - and
`push_to_hub="false"` published the finished dataset to the Hub. Both are now checked on
the shared `utils.boolean_flag_error` domain through the new
`simulation.recording.dataset_recording_posture_error`, on all three backends, ahead of the
lerobot-extra probe and ahead of anything on disk.
