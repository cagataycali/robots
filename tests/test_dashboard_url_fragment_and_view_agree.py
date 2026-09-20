"""The address bar and the screen name the same tab, whichever way the tab was entered.

`show()` writes `location.hash` on every tab change, so the page fills the
browser's history with entries it owns. Only the initial load read that
fragment back: there was no `hashchange` listener, so a fragment that changed
on an open page - a bookmarked `#agent` pasted into the tab, a link, Back or
Forward through the entries `show()` itself wrote - moved the URL and left the
view where it was. Measured in a browser on the pre-fix files, Back from Fleet
to `#agent` showed Fleet under the address `#agent`; an operator reaching for
the Agent tab read the Fleet page and the highlighted tab agreed with the page,
not with the URL they asked for.

The dispatch that enters a view is now one function, `route()`, and the three
ways in - a tab click, the loaded URL's fragment, a fragment change - all call
it. These cells read the shipped `app.js` for that shape, so the rule holds
without a browser in the suite.
"""

from __future__ import annotations

import pathlib
import re

APP_JS = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "static" / "app.js"

#: Every view loader, keyed by the view whose entry must call it.
LOADERS = {"fleet": "loadFleet()", "sim": "loadSim()", "agent": "loadAgent()", "settings": "loadSettings()"}


def _function_body(source: str, name: str) -> str:
    """The text of ``function <name>(...) { ... }``, braces balanced."""
    start = source.index(f"function {name}(")
    depth = 0
    for i in range(source.index("{", start), len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after function {name}")


class TestTheFragmentAndTheViewAgree:
    def test_one_router_is_the_only_place_a_view_is_entered(self) -> None:
        """The view-to-loader dispatch lives once, in route(); a second copy drifts from it."""
        source = APP_JS.read_text(encoding="utf-8")
        assert "function route(" in source, "no router: each entry point carries its own dispatch"
        router = _function_body(source, "route")
        for view, call in LOADERS.items():
            dispatch = re.compile(rf'===\s*"{view}"\)\s*{re.escape(call)}')
            assert dispatch.search(router), f"route() enters {view} without loading it"
            assert len(dispatch.findall(source)) == 1, (
                f"the {view} dispatch is written twice, so one copy can be updated and the other left behind"
            )

    def test_a_fragment_change_routes_to_the_view_it_names(self) -> None:
        """A hashchange - bookmark, link, Back, Forward - enters the view the fragment names."""
        source = APP_JS.read_text(encoding="utf-8")
        listener = re.search(r'window\.addEventListener\("hashchange".*?\n\}\);', source, re.S)
        assert listener, "no hashchange listener: the URL can name a tab the page is not showing"
        body = listener.group(0)
        assert "route(" in body, "the hashchange listener does not enter the view"
        assert "shownView()" in body, (
            "the listener does not skip the view already shown, so show()'s own hash write re-enters it"
        )

    def test_the_tab_click_and_the_load_share_that_router(self) -> None:
        """The two older entry points call route() rather than repeating its body."""
        source = APP_JS.read_text(encoding="utf-8")
        click = source[source.index('$("#tabs").addEventListener("click"') :]
        click = click[: click.index("\n});") + 4]
        assert "route(" in click, "a tab click does not go through the router"
        assert "route(" in _function_body(source, "boot"), "the loaded URL's fragment does not go through the router"
