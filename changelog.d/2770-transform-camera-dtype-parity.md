### Fixed: a transformed dataset keeps the dtype each source camera declared

A LeRobot dataset declares each camera's storage per feature: one
`observation.images.<cam>` can be `dtype="video"`, encoded into an MP4 under `videos/`, while
another is `dtype="image"`, with its frames in the data parquet. `LeRobotDataset.create` accepts
that declaration and reads both streams back, so it is a legal source. The output side has one
knob for the whole dataset -- `DatasetRecorder.create` takes a single `use_videos` flag that
decides every camera's dtype -- and `_SourceDataset` derived that flag by assigning it once per
camera inside the loop that reads the camera shapes. It therefore held whichever camera the
source's feature dict ended on.

A two-camera source declaring one video stream and one image stream transformed with
`status="success"` while the output stored a camera a different way, and the direction depended
only on the order the features were declared in. Declaring the video stream first flattened it
into still frames, so a source with one MP4 produced an output with none; declaring it last
promoted the image column to video, so the same source produced two.

`create_output_recorder` documents itself as "the SOURCE schema (parity by construction)", schema
parity is the first acceptance criterion of the transform round-trip suite, and
`docs/data/transforms.md` contract item 1 promises a generated episode is the same trajectory
rendered differently. A stream stored a different way is none of those.

The flag is now derived from the whole camera set, and a source whose cameras disagree is refused
by `_SourceDataset.open`, naming each camera and the dtype it declared so an operator knows which
stream to convert. That is the disposition `open` already takes for a source declaring a feature
the pass-through cannot preserve, and for the same reason: the output recorder declares one dtype
for every camera, so a mixed source cannot be reproduced, and writing it anyway would alter a
column the contract promises to carry through. Re-encoding to one dtype instead would be a quieter
version of the same answer -- coercing to `image` also flattens the video streams of sources that
work today.

Nothing changes for a source whose cameras agree, whichever dtype that is: an all-video recording
still transforms to video and an all-image one still transforms to images, because a homogeneous
set already derived the right flag whichever camera the loop ended on. Every camera's dtype is now
recorded rather than only the last one, so the refusal can report the disagreement it found.
