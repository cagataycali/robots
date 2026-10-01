"""The dashboard is the fleet on the mesh, its cards, and the agent that drives them.

The record and train screens and the in-process Sim tab left together with the
agent console's seven ``sim_*`` tools: a robot the dashboard shows or moves is a
mesh peer, or it is not on this page. These cells read the page's source and the
console the way a reviewer would, so a screen or a tool that grows back is caught
before it reaches an operator. The backend routes those screens called
(``/api/record``, ``/api/training``, ``/api/sim``) stay for API users; only the
page and the agent stopped calling them.
"""

from __future__ import annotations

import pathlib
import re

FRONTEND = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "frontend" / "src"
STATIC = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "static"

#: The screens that left, by component file name and by the fragment that used to open them.
RETIRED_SCREENS = {"SimTab": "sim", "RecordPanel": "record", "TrainingTab": "training"}

#: The agent console's in-process simulation tools that left with them.
RETIRED_SIM_TOOLS = ("robots", "sim_sessions", "sim_start", "sim_state", "sim_set_joints", "sim_reset", "sim_stop")


def _panels() -> list[str]:
    source = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    block = re.search(r"const PANELS[^=]*=\s*\[(.*?)\]", source, re.S)
    assert block, "App.tsx no longer declares PANELS; these cells need the router's list"
    return re.findall(r"'([a-z]+)'", block.group(1))


def test_the_page_has_no_record_train_or_sim_screen() -> None:
    components = {p.stem for p in (FRONTEND / "components").glob("*.tsx")}
    assert not (components & set(RETIRED_SCREENS)), "a retired screen's component is back in the tree"
    panels = _panels()
    assert not (set(panels) & set(RETIRED_SCREENS.values())), f"the router still routes to a retired screen: {panels}"
    assert {"settings", "activity", "devices", "estop", "help"} <= set(panels)


def test_the_fleet_bar_offers_no_record_train_or_sim_chip() -> None:
    source = (FRONTEND / "components" / "FleetBar.tsx").read_text(encoding="utf-8")
    for prop in ("onRecord", "onTraining", "onSim", "recordMock"):
        assert prop not in source, f"FleetBar still takes {prop}"
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    for name in RETIRED_SCREENS:
        assert name not in app, f"App.tsx still mounts {name}"


def test_the_shipped_bundle_calls_no_sim_or_training_route() -> None:
    """The built page is what an operator loads; the source alone could be ahead of it."""
    generated = (FRONTEND / "lib" / "bundleRoutes.generated.ts").read_text(encoding="utf-8")
    routes = re.findall(r"'(/api/[^']*)'", generated)
    for route in routes:
        assert not route.startswith(("/api/sim", "/api/training", "/api/collect", "/api/replay")), (
            f"the page still calls {route}, a route of a screen that left"
        )
    app_js = (STATIC / "app.js").read_text(encoding="utf-8", errors="replace")
    for route in ("/api/sim", "/api/training/submit", "/api/collect"):
        assert route not in app_js, f"static/app.js was not rebuilt: it still calls {route}"


def test_the_agent_console_has_no_in_process_sim_tool() -> None:
    from strands_robots.dashboard import agent_console

    for name in RETIRED_SIM_TOOLS:
        assert not hasattr(agent_console, name)
    assert agent_console.expected_tool_names(None) == ["emergency_stop"]
    assert not hasattr(agent_console, "MotionGate"), "the sim gate left with the sim tools"
