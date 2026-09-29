### Added: `--mesh-listen`, the dashboard as the mesh anchor

`strands_robots dashboard --mesh-listen tcp/0.0.0.0:7447` sets `ZENOH_LISTEN`
before the mesh session opens, so the dashboard is the first node robots on the
LAN attach to. The header shows the line they need
(`ZENOH_CONNECT=tcp/<lan-ip>:7447`, click to copy) and `/api/mesh/config`
carries it as `anchor_hint`. A malformed endpoint is refused before anything
starts.
