### Fixed: the dashboard page can no longer choose the host that receives `OPENAI_API_KEY`

`OPENAI_BASE_URL` was page-writable, so a dashboard session could point the
OpenAI client (and the stored `OPENAI_API_KEY` it sends) at any host. It is
still shown in the env view, but is now set on the host only, like
`STRANDS_DASH_RECORD_CRUMB` and `STRANDS_ROBOTS_VIDEO_ROOT`.
