### Fixed: Newton and Isaac stop rendering cameras for a recording that keeps none

`get_observation(skip_images=True)` during a `start_recording(cameras=[])`
session now honours the skip on Newton and Isaac, as it already did on MuJoCo:
the dataset declares no image column, so the frames were rendered only to be
dropped. A recording that keeps cameras still turns the hint off on every
backend. The three backends now ask one shared
`DatasetRecordingMixin._recording_keeps_images()` instead of carrying their own
copy of the check.
