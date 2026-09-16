### Fixed: `replay_episode` on a missing dataset names what was checked

A typo'd or never-recorded `repo_id` surfaced as a raw `huggingface_hub`
404 (request id, `repo_type` advice, gated-repo paragraph) after a network
round trip. The refusal now names the directory checked, the Hub verdict,
the datasets beside it on disk, this session's most recent recording and
the remedy (`root=` or record one). Offline reads as "Hub could not be
reached"; other errors keep their own text.
