"""The fleet view's IoT words are computed by three pure helpers the page imports.

``cameraPath.ts`` captions a tile with ``S3`` when the bridge fetched the frame
through a reference and with the publisher-to-dashboard latency once it is a WAN
number; ``pingVerdict.ts`` turns the ping route's verdict word into the label
beside the button; ``RegistryCard.lastSeenLabel`` ages a Thing's last presence.
The cells run the TypeScript under Node (tests/_dashboard_frontend) and pair with
static source cells so the page keeps importing the helpers it is graded on.
"""

from __future__ import annotations

import re

import pytest

from tests._dashboard_frontend import FRONTEND_SRC, requires_node, run_frontend

_COMPONENTS = FRONTEND_SRC / "components"


@requires_node
class TestCameraPath:
    def test_an_s3_frame_is_captioned_and_a_lan_frame_stays_quiet(self) -> None:
        got = run_frontend(
            """
const m = await import('./cameraPath.ts')
out({
  s3: m.cameraPathLabel({ via: 's3', latency_ms: 420 }),
  inline: m.cameraPathLabel({ via: 'inline', latency_ms: 12 }),
  older: m.cameraPathLabel(undefined),
  wan: m.cameraLatencyLabel({ via: 's3', latency_ms: 420 }),
  lan: m.cameraLatencyLabel({ via: 'inline', latency_ms: 12 }),
  iot: m.cameraLatencyLabel({ via: 'inline', latency_ms: 92 }),
  slow: m.cameraLatencyLabel({ via: 's3', latency_ms: 12_400 }),
  none: m.cameraLatencyLabel({ via: 's3', latency_ms: null }),
  nan: m.cameraLatencyLabel({ via: 's3', latency_ms: Number.NaN }),
  threshold: m.LATENCY_SHOWN_MS,
})
"""
        )
        assert got["s3"] == "S3" and got["inline"] == "" and got["older"] == ""
        assert got["wan"] == "420 ms" and got["lan"] == "" and got["slow"] == "12 s"
        assert got["iot"] == "92 ms"
        assert got["none"] == "" and got["nan"] == ""
        assert got["threshold"] == 50


@requires_node
class TestPingVerdict:
    @pytest.mark.parametrize(
        ("ping", "label"),
        [
            ({"thing": "a", "verdict": "answered", "latency_ms": 91.4}, "answered in 91 ms"),
            ({"thing": "a", "verdict": "offline", "latency_ms": 80}, "offline (broker 404 in 80 ms)"),
            ({"thing": "a", "verdict": "forbidden"}, "forbidden for this operator"),
            ({"thing": "a", "verdict": "silent", "latency_ms": 95}, "delivered, no answer"),
            ({"thing": "a", "verdict": "unavailable"}, "no direct send on this backend"),
            ({"thing": "a", "verdict": "refused"}, "refused"),
            ({"thing": "a", "verdict": "error", "reason": "direct send throttled"}, "error: direct send throttled"),
            ({"thing": "a", "verdict": "pending", "pending": True}, "pinging\u2026"),
        ],
    )
    def test_every_verdict_word_has_a_label(self, ping: dict, label: str) -> None:
        import json

        got = run_frontend(
            f"""
const m = await import('./pingVerdict.ts')
out({{ label: m.pingLabel({json.dumps(ping)}), empty: m.pingLabel(undefined) }})
"""
        )
        assert got["label"] == label
        assert got["empty"] == ""


class TestThePageImportsWhatIsGraded:
    def test_the_camera_tile_reads_both_caption_helpers(self) -> None:
        src = (_COMPONENTS / "CameraTile.tsx").read_text(encoding="utf-8")
        assert "from '../lib/cameraPath'" in src
        assert "cameraPathLabel(meta)" in src and "cameraLatencyLabel(meta)" in src

    def test_the_registry_card_reads_the_ping_label_and_takes_the_ping(self) -> None:
        src = (_COMPONENTS / "RegistryCard.tsx").read_text(encoding="utf-8")
        assert "from '../lib/pingVerdict'" in src
        assert "pingLabel(ping)" in src
        assert "disabled={!!ping?.pending}" in src

    def test_the_reach_chip_sits_after_the_host_so_a_name_is_never_truncated(self) -> None:
        src = (_COMPONENTS / "RobotCard.tsx").read_text(encoding="utf-8")
        host = src.index('className="host"')
        chip = src.index("className={`reachchip ${peer.reach}`}")
        assert host < chip, "the reach chip must render after the host, at the right edge of the card head"
        assert re.search(r"peer\.reach === 'lan' \|\| peer\.reach === 'iot' \|\| peer\.reach === 'both'", src)

    def test_the_page_offers_ping_only_where_the_server_says_it_can_address_a_thing(self) -> None:
        src = (FRONTEND_SRC / "App.tsx").read_text(encoding="utf-8")
        assert "registry?.ping_available ? pingThing : undefined" in src
        assert "/api/robots/${encodeURIComponent(name)}/ping" in src
