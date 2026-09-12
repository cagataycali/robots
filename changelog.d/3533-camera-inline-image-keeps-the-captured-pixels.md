### Fixed: `lerobot_camera` returns the captured frame's own pixels

A `capture` / `capture_batch` result carries the saved file *and* an inline copy
of the frame for the model to look at. The inline copy was encoded as JPEG for
every `format` the encoder's branch did not name - so a caller who chose a
lossless container got a lossy image back, silently, under `status="success"`.

Measured against a captured 8x6 frame, `format="bmp"` - one of the three the
tool's own docstring lists - wrote a real BMP to disk and handed back a JPEG
whose pixels differed from the frame by up to 213 of 255. `format="tiff"`,
`"webp"` and `"gif"` reached the same fallback: OpenCV wrote each container
faithfully, and the half of the result an agent actually reads was the half that
discarded the frame.

JPEG is now the answer to a JPEG request and to nothing else; every other
spelling is carried back as PNG, which is lossless and which the Converse API
accepts. `format="jpg"` and `format="png"` are unchanged, and no request that
worked before is refused - the file on disk is written exactly as it was.
