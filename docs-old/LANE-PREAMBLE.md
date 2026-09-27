# Lane preamble (read fully before your first edit)

You are one lane of a from-scratch rewrite of the strands-robots documentation.
Read `/Users/cagatay/robots-docs/docs-old/DESIGN.md` first. It is the spec. Sections 4, 5 and 6
(architecture rules, voice, visual) are hard rules; a page that breaks them is not done.

## Where you work

- Worktree: `/Users/cagatay/robots-docs` (branch `docs/from-scratch-20260927`, off `strands-labs/robots` main `1d12a659`).
- The package source is right there: `strands_robots/`. Read it. The docs describe THIS commit, nothing else.
- The old site is preserved under `docs-old/` (pages in `docs-old/**`, nav in `docs-old/mkdocs.old.yml`). Mine it for facts, never copy prose. Every sentence you keep must be re-verified against the source.
- Docs venv for building: `.venv-docs/bin/mkdocs`. Full package venv for verifying fences: `/Users/cagatay/robots/.venv/bin/python` (has mujoco, lerobot, etc.). Run fences with `PYTHONPATH=/Users/cagatay/robots-docs` so they import this checkout.
- `git` in the worktree: commit your own paths only, prefix `docs(<lane>): ...`. Never `git add -A`. Never push. Never touch `mkdocs.yml` unless your lane owns a listed part of it (D5 appends to `redirect_maps` only).

## Hard rules

1. Edit only the paths your lane owns (DESIGN.md section 7). Anything else you need goes in `docs-old/handoffs/<lane>.md` as a request to the coordinator.
2. No em dashes (—) or en dashes (–) anywhere. Check with `rg -n "—|–" <your paths>` before each commit.
3. Under 900 words per page (`wc -w`). Split rather than exempt.
4. Every `python` fence that is meant to run must run. Mark runnable fences with the info string `python` and non-runnable illustrations with `python title="sketch"`. Verify runnable fences with the full venv before you commit. A fence that needs hardware or a GPU is written in `bash`/`python title="sketch"` and says what it needs in the sentence above it.
5. Numbers about the codebase come from the numbers hook token `{{numbers:<key>}}` (coordinator owns the key list; request new keys in your handoff) or from a generated table. Do not type counts by hand.
6. Names of robots, policies, tools, env vars, refusal codes are exact identifiers from the source. Grep before you write.
7. Every page starts with one paragraph that says what the reader has at the end of it. Then code.
8. Link with relative paths to `.md` files (mkdocs strict resolves them). A link to a page another lane is writing uses the path in DESIGN.md section 2; if it does not exist yet, still write it (the coordinator's final build catches misses).
9. Robot viewer: `<robot-viewer name="<registry_name>"></robot-viewer>` on its own line inside a page, with `markdown` attribute not needed. Do not autoload attributes; the coordinator owns the component.
10. Do not write tests. Do not edit `tests/`. Do not edit `strands_robots/`.

## Working method

Read broadly first (the whole subtree you document, `rg` for the public functions and their docstrings), then write the page outline as H2s, then fill. Prefer one table over three paragraphs. Prefer a generated table over a hand-written one; if a table's rows come from the source (providers, drivers, env vars), write a hook under `docs/hooks/` that emits it and register it in your handoff for the coordinator to wire.

Hooks are plain mkdocs hooks or `mkdocs-gen-files` scripts: pure Python, read files, never import `strands_robots` (the docs build runs without the package installed).

## Done means

- All your pages exist at the DESIGN.md paths, build under `mkdocs build --strict` when the coordinator wires them (test locally: temporarily add them to a scratch copy of mkdocs.yml, `mkdocs build --strict -f /tmp/scratch.yml`, then delete the scratch copy).
- Fences verified, dashes zero, budget under 900, links relative.
- `docs-old/handoffs/<lane>.md` written: what you shipped (paths), what you verified and how, coordinator requests (nav entries in order, hook registrations, numbers keys), open questions.
- Committed on the branch with your prefix.
- Your last message is a short table: page, words, fences run, status.
