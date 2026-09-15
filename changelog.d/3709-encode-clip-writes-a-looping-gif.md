### Fixed: `encode_clip` writes a GIF that loops

`encode_clip` is the only GIF writer in the package, and its Pillow branch never
wrote the looping extension block, so every GIF it produced played once and
froze on its last frame. It now passes `loop=0`, which is what the animated GIFs
this repo ships carry and what callers encoding a *clip* asked for. MP4 is
unaffected - looping is a GIF container feature.
