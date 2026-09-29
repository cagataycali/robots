### Docs: the site moves when something happens

The docs site had no motion beyond a hover colour: pages popped, the theme
toggle flashed, the proof numbers sat still. Material's instant navigation is
now on (the content column fades in after a swap, never on first paint), the
landing's numbers count up once when seen, a copy pill says "copied", the "On
this page" bar travels to the active entry, the page head's rule draws in and
the theme toggle cross-fades where the browser has the View Transitions API.
All of it collapses to its end state under `prefers-reduced-motion`, pinned by
`tests/test_docs_motion_layer.py` (#4204), which also fails any local script that does
not re-run on `document$` (the landing's robot picker was wired once at load
and would have died on the first instant navigation).
