### Fixed: the dashboard tab the address names is the tab the page shows

The page writes `location.hash` on every tab change, filling the browser's
history with entries it owns, but only the initial page load read a fragment
back. A bookmarked `#agent` pasted into an open tab, a link, or Back and
Forward through those entries moved the address and left the view where it was:
Back from Fleet to `#agent` showed Fleet, with the tab highlight following the
page rather than the URL. One `route()` is now the only place a view is
entered, and a `hashchange` listener enters the view the fragment names - while
skipping the view already shown, so the page's own hash write does not re-enter
it and a tab click still loads once.
