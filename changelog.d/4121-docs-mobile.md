### Docs: the site reads on a phone, and the robot viewer works under a finger

The rewritten docs site (#4112) was checked on a desktop. On an iPhone or a
Pixel the landing page laid out at 652 px and zoomed out, the install line
wrapped mid word, generated tables broke identifiers across lines, and the
MuJoCo viewer autoloaded 27 MB of engine and meshes, showed the robot as a
270 px strip, clipped two of its pills, and opened the joints panel over the
robot where a drag hit the panel instead of the camera. Grid tracks are now
``minmax(0, 1fr)``, code-only table cells stay on one line while the table
scrolls in Material's wrapper, the viewer is a poster on phones until Load 3D
is tapped, its stage is 4:5 under 40em with 40 px pills and a bottom sheet
for joints, fullscreen falls back to a fixed layer where
``requestFullscreen`` is missing (iPhone Safari), touch targets are 40 to
44 px, robot cards are 2 columns on a tablet and 1 on a phone, and reduced
motion is honoured (#4121).
