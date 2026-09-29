"""Checkpoint autocomplete that leads to a run the card can show.

* the search ranks checkpoints that name the selected robot first and marks them, so
  the menu's first rows are the ones trained on the robot being driven (#4212);
* the picker is a keyboard-navigable combobox (roles and the active-row rule are
  pinned on the shipped bundle, which is the text a reviewer reads);
* a policy running on a simulation peer shows as running: sim peers publish
  ``robots.<name>.active`` and no ``task.status`` (#4182), and both the card and the
  voice agent's ``fleet peers`` now read that flag.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.dashboard import checkpoints

_APP_JS = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "static" / "app.js"

# ─────────────────────────────────────────── ranking ───────────────────────


@pytest.mark.parametrize(
    ("robot", "expected"),
    [
        ("so101", {"so101", "so-101", "so_101"}),
        ("unitree_go2", {"unitree_go2", "unitree-go2", "unitreego2", "unitree", "go2", "go-2", "go_2"}),
        ("franka_panda", {"franka_panda", "franka-panda", "frankapanda", "franka", "panda"}),
        ("", set()),
        ("g1", set()),  # too short on its own: it would match half the Hub
    ],
)
def test_robot_tokens_cover_the_spellings_hub_authors_use(robot: str, expected: set[str]) -> None:
    assert set(checkpoints.robot_tokens(robot)) >= expected
    assert all(len(t) >= 3 for t in checkpoints.robot_tokens(robot))


ROWS = [
    {"repo_id": "lerobot/smolvla_base", "tags": ["lerobot", "robot_type:so100"]},
    {"repo_id": "robotfuel/act_so101_t16b", "tags": []},
    {"repo_id": "someone/pi0_koch", "tags": ["robot_type:koch"]},
    {"repo_id": "x/act-so-101-cube", "tags": []},
]


def test_rank_for_robot_puts_the_robots_rows_first_and_marks_every_row() -> None:
    ranked = checkpoints.rank_for_robot(list(ROWS), "so101")
    assert [r["repo_id"] for r in ranked] == [
        "robotfuel/act_so101_t16b",
        "x/act-so-101-cube",
        "lerobot/smolvla_base",
        "someone/pi0_koch",
    ]
    assert [r["robot_match"] for r in ranked] == [True, True, False, False]


def test_rank_for_robot_is_stable_within_each_half() -> None:
    ranked = checkpoints.rank_for_robot(list(ROWS), "koch")
    assert [r["repo_id"] for r in ranked] == [
        "someone/pi0_koch",
        "lerobot/smolvla_base",
        "robotfuel/act_so101_t16b",
        "x/act-so-101-cube",
    ]


def test_rank_for_robot_matches_on_tags_too() -> None:
    ranked = checkpoints.rank_for_robot(list(ROWS), "so100")
    assert ranked[0]["repo_id"] == "lerobot/smolvla_base" and ranked[0]["robot_match"] is True


def test_no_robot_means_no_ranking_and_no_marks() -> None:
    assert checkpoints.rank_for_robot(list(ROWS), None) == ROWS
    assert checkpoints.rank_for_robot(list(ROWS), "") == ROWS


def test_search_ranks_the_merged_rows_and_reports_the_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "trained_checkpoints", lambda q: [])
    monkeypatch.setattr(
        checkpoints, "local_checkpoints", lambda q: [{"repo_id": "lerobot/smolvla_base", "local": True, "tags": []}]
    )
    monkeypatch.setattr(
        checkpoints,
        "hub_search",
        lambda q, limit=12: ([{"repo_id": "robotfuel/act_so101_t16b", "local": False, "tags": []}], None),
    )
    monkeypatch.setattr(checkpoints, "hf_auth_state", lambda: {"authenticated": False})
    out = checkpoints.search("act", limit=10, robot="so101")
    assert out["robot"] == "so101"
    assert [r["repo_id"] for r in out["results"]] == ["robotfuel/act_so101_t16b", "lerobot/smolvla_base"]
    assert checkpoints.search("act", limit=10)["robot"] is None


def test_search_route_passes_the_robot_hint() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app

    seen: dict[str, Any] = {}

    def fake_search(q: str, limit: int, robot: str | None = None) -> dict[str, Any]:
        seen.update(q=q, limit=limit, robot=robot)
        return {"query": q, "robot": robot, "results": [], "total_matched": 0, "hub_problem": None, "hf_auth": {}}

    app = create_app()
    with TestClient(app) as client, pytest.MonkeyPatch.context() as mp:
        mp.setattr(checkpoints, "search", fake_search)
        assert client.get("/api/checkpoints/search?q=act&limit=5&robot=so101").status_code == 200
        assert seen == {"q": "act", "limit": 5, "robot": "so101"}
        client.get("/api/checkpoints/search?q=act")
        assert seen["robot"] is None


# ─────────────────────────────────────────── the picker, as shipped ────────


def test_the_shipped_picker_is_a_combobox_with_an_active_row() -> None:
    js = _APP_JS.read_text(encoding="utf-8")
    picker = js[js.index("function CheckpointPicker") :]
    picker = picker[: picker.index("\nfunction ", 1) if "\nfunction " in picker[1:] else None]
    for needle in (
        'role: "combobox"',
        'role: "listbox"',
        'role: "option"',
        '"aria-activedescendant"',
        '"aria-expanded"',
        "nextActive(",
    ):
        assert needle in picker, needle
    for key in ("ArrowDown", "ArrowUp", "Home", "End"):
        assert f'"{key}"' in js  # nextActive's switch, bundled from lib/checkpointRobot.ts
    assert "&robot=" in picker  # the robot hint reaches the server
    assert "policy-fit?repo_id=" in picker  # the fit pill asks the fit route, not a guess


def test_the_shipped_card_reads_a_sim_peers_activity_as_running() -> None:
    """#4182: `reportedTaskStatus` falls back to robots.<name>.active; both readers use it."""
    js = _APP_JS.read_text(encoding="utf-8")
    assert js.count("simActivityStatus(") >= 3  # definition + reportedTaskStatus + peerStatusFields
    fn = js[js.index("function simActivityStatus") :]
    fn = fn[: fn.index("\n}") + 2]
    assert '"running"' in fn and '"idle"' in fn
    assert re.search(r'split\("__"\)\[1\]', fn), "a child peer is asked about its own robot only"


# ─────────────────────────────────────────── voice: fleet peers ────────────


def test_voice_fleet_peers_reads_a_sim_rollout_as_running() -> None:
    pytest.importorskip("strands")
    from strands_robots.dashboard.voice import make_fleet_tool

    class Bridge:
        def snapshot(self) -> dict[str, Any]:
            return {
                "peers": {
                    "lane-so101__so101": {
                        "presence": {"robot_type": "sim"},
                        "state": {"joints": {"1": {}}, "robots": {"so101": {"active": True}}},
                    },
                    "idle-sim__so101": {
                        "presence": {"robot_type": "sim"},
                        "state": {"robots": {"so101": {"active": False}}},
                    },
                    "arm-1": {
                        "presence": {"robot_type": "robot"},
                        "state": {"task": {"status": "running", "instruction": "wave"}},
                    },
                }
            }

    text = make_fleet_tool(Bridge())(action="peers")["content"][0]["text"]
    lines = {line.split(":")[0].strip("- "): line for line in text.splitlines() if line.startswith("- ")}
    assert "task=running" in lines["lane-so101__so101"]
    assert "task=idle" in lines["idle-sim__so101"]
    assert "task=running instruction='wave'" in lines["arm-1"]
