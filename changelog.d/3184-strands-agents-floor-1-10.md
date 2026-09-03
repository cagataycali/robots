### Changed: the `strands-agents` floor rises to 1.10.0, the release that ships `BeforeToolCallEvent`

`strands.hooks.BeforeToolCallEvent` is first exported by `strands.hooks` in
strands-agents **1.10.0**, measured against every released 1.x wheel. The
declared floor was `>=1.7.0`, so five releases (1.7.0, 1.7.1, 1.8.0, 1.9.0,
1.9.1) satisfied packaging without shipping the name: on 1.9.1 a
`strands-robots[dashboard]` install raises `ImportError: cannot import name
'BeforeToolCallEvent' from 'strands.hooks'` the moment
`strands_robots.dashboard.agent_hitl` is imported, so the release range decided
whether the human-in-the-loop motion gate existed to be wired at all. The floor
now states that capability in `project.dependencies`, in the `[ollama]` extra
that re-declares it, and in `uv.lock`'s transcription of both.
