### Fixed: `sample_frames` labels each image block with its camera, so a per-view judgement can name its view

`sample_frames(include_images=True)` handed a multimodal judge a flat run of
`n_frames x n_cameras` images and one leading sentence stating the grouping
("position-major, cameras sorted (...)"). Naming a view then required applying
that rule to an image's ordinal, and a rule stated at a distance from the images
does not bind. Measured on a three-camera recording with one view fully blocked
(0 object pixels in every sampled frame of that view, 1.2-2.1% of frame in the
other two), naming the blocked camera from the payload scored 15/40, while the
same model scored 30/30 on the identical frames asked one at a time - so the
frames carried the answer and the payload's structure was what lost it. Two of
the eight prompt variants answered the same camera whichever view was actually
blocked, which is how a single-scene check reads 5/5 on one dataset and 1/5 on
another.

Every image block is now immediately preceded by a text block naming its camera
and `frame_index`, which also makes the label a join key onto the state row for
the same frame in `samples`. The same measurement scores 40/40, unchanged across
option order and question position and on both datasets, for +96 prompt tokens
(+4.0%). The label precedes its image because that is what binds: with the
labels moved after their images and nothing else changed - same bytes, same
token count - the score is 0/40, and every wrong answer is the camera named by
the label that then preceded the blocked image, so a text block is read as a
caption of the image that follows it. Spelling the whole block map out in one
leading text block instead scores 17/40 at more tokens than the fix, so what the
label buys is adjacency, not more words.

`sample_frames` is the only surface in the package that emits more than one image
block in a single payload; the leading grouping sentence is unchanged.
`docs/data/episode-labels.md` no longer attributes an unnamed `camera_occlusion`
view to judge capability.
