### Fixed: recorded camera features are declared HWC, like every lerobot v3 dataset

`DatasetRecorder` was the only writer declaring cameras CHW - shape
`(3, H, W)`, names `[channels, height, width]`. It now declares them through
lerobot's own `hw_to_dataset_features`: `(H, W, 3)`, `[height, width, channels]`,
the layout of lerobot's record path and of every published v3 dataset. Schema
only: frames, decoding and training are unchanged (training transposes by
names), and datasets recorded as CHW still load, train and resume. The resume
schema check read the camera shape positionally as CHW and refused to resume
any HWC dataset ("resolution differs: on-disk=(640, 3)"); it now reads by names.
