### Changed: the dashboard wears the docs design

The operator dashboard now uses the documentation site's design tokens
(`docs/stylesheets/extra.css`): a white or black page, black or white ink and
card outlines, the one Strands green for links, live signals and filled
controls, the docs' yellow for warnings, an AA red kept for physical danger
only, JetBrains Mono for headings, labels and pills over Space Grotesk text,
8px corners and outlined 999px pills. The frosted-glass panes, ambient
gradients, glows and drop shadows are gone; an edge is a line. The header is the
docs header: the pixel STRANDS wordmark linking to strandsagents.com, then
`/robots`.

Two schemes, `paper` and `dark`, follow the OS by default; a chip in the fleet
bar switches them and the choice is remembered on the device, the same
semantics as the docs' palette toggle. The two typefaces ship with the bundle
as variable woff2 files (OFL 1.1, listed in `static/vendor/NOTICE`) so a
dashboard on a robot LAN with no internet renders in them. Reduced-motion and
forced-colours rules are unchanged.
