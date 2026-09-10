### Fixed

- `StreamingDatasetReader.open(..., drop_videos=True)` now really skips video decode. lerobot's `StreamingLeRobotDataset` decodes every key in `meta.video_keys` whether or not it appears in `delta_timestamps`, so the previous implementation (strip camera deltas) still hit torchcodec on the first frame - the documented "works without a torchcodec wheel" was false. The reader now hides the video features on its instance metadata (memory only). The monkeypatched test that let this ship is replaced by one that runs the real `StreamingLeRobotDataset` on a two-frame fixture.
