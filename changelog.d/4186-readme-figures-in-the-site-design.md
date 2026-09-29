### Docs: the README figures wear the strandsagents.com design

`docs/assets/hero_loop.svg`, `architecture_flow.svg` and `mesh_network.svg`
still drew the old site: neon green with cyan, radial glows, a glass panel
and the system sans, while the docs had moved to the strandsagents.com design
in #4112. The three are redrawn from the same tokens the site uses (black or
white surface, one Strands green, JetBrains Mono over Space Grotesk, outlined
pills, halftone band, the terminal card), with the typefaces subset and
embedded so GitHub renders them, a light variant through
`prefers-color-scheme`, and motion that stops under `prefers-reduced-motion`.
No robot count is typed into a drawing. README alt texts describe the new
figures (#4186).
