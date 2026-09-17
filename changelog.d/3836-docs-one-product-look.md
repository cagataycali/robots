### Documentation

- **docs site**: the look is one stylesheet again. `docs/stylesheets/extra.css`
  was 489 lines of per-element overrides (plus a 15-line card sheet and a
  422-line orphan copy under `docs/assets/` that nothing loaded); it is now 199
  lines that set Material's own custom properties - a paper light scheme, a true
  dark scheme, one accent declared once per scheme, 16px/1.65 prose on a 38rem
  measure inside a 76rem page so a coverage matrix is no longer squeezed into a
  688px column. On a 390px viewport the 18 tables that scrolled sideways (worst
  177px) now stack into one card per row, each cell labelled with its column
  header; every content image loads lazily, and demo GIFs cap at 480px without
  overflowing a phone. The sidebar no longer renders every page of every section
  at once (2,308px of nav on the home page, 773px now). Lighthouse accessibility
  on the home page goes 93 to 100: the code surface is light enough for
  Material's syntax tokens to keep 4.5:1, and the theme's search overlay - a
  `role="dialog"` with no accessible name - is named.
