### Changed: the dashboard works on a phone and a tablet

On a 390x844 phone the page used to scroll 459 px sideways, its header was
221 px tall, and STOP ALL sat at x=707, past the right edge of the screen. The
layout now has three breakpoints (640, 900, 1200 px). Below 640 px the bar
shows the wordmark, the LIVE pill and one menu button that opens a bottom
sheet, and STOP ALL is a fixed red 44 px pill at the bottom left that is
never inside the menu. Help, e-stop, consent, run-confirm and robot detail
open as full-width bottom sheets. Tablets get two columns and a scrollable
chip strip. Type uses a rem scale with nothing under 11 px.

Measured on fleet, menu, devices, activity, settings, help and the agent dock
at 390 px, paper and dark: 0 px sideways scroll, no tap target under 44 px,
STOP ALL on screen in every view, and axe-core (wcag2a/aa/21aa and
best-practice) reports 0 serious or critical findings, where main had up to 10.
At 1440 px the desktop layout keeps its intent. Routes, fetches, auth,
consent and e-stop behaviour are unchanged, and no dependencies were added.
