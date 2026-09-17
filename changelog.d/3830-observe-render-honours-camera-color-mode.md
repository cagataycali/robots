### Fixed: a real arm's `render` saves the colour the camera saw

`Robot(mode="real", cameras={...})` accepts every field of lerobot's camera
config, `color_mode` included, and a camera configured `color_mode="bgr"`
delivers frames in OpenCV's own channel order. The observe path encoded every
frame with `COLOR_RGB2BGR` regardless, so a BGR camera's frame was converted a
second time and red and blue were transposed: a photograph of a red object was
written - and handed to the model as an image block - as pure blue, under a
result reporting success. The frame is now encoded in the order its camera
delivers, read off the camera's own `color_mode`, in the OpenCV and the Pillow
encoder alike.
