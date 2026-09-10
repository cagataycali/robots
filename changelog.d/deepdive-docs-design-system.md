### Changed: docs theme - one accent variable, paper light / true dark, no glass

`docs/stylesheets/extra.css` is rewritten (11.9 KB -> 5.8 KB): a single
`--sr-accent` token drives Material's primary/accent colours, hairline
borders, one radius, system font stack (`font: false`, no Google Fonts
request), motion <= 180 ms with a `prefers-reduced-motion` kill switch. The
glassmorphic blur/neon-green rules and the dead duplicate
`docs/assets/extra.css` (10.3 KB, referenced nowhere) are removed.
