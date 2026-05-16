# Docs site rewrite — autonomous build plan

**Branch:** `docs/mkdocs-site` (off `main`, fork: `cagataycali/robots`)
**Goal:** ship a complete MkDocs Material site that documents the *current* `strands-robots` codebase end-to-end, so a developer can land here and be productive in 15 minutes.

The visual scaffolding (`mkdocs.yml`, `docs/assets/`, `docs/stylesheets/`) was lifted from PR #40 and lives on this branch. **Every other `.md` page must be written from scratch against the current main code.** The PR40 content is stale (referenced removed modules).

## Hard rules

1. **Source of truth is the code on this branch.** Open files. Read docstrings. Read tests. Quote real APIs. Never hallucinate parameters, never copy from PR40 prose.
2. **Every code block must work.** Run a smoke check: `python3 -c "<the code block>"` for non-network-bound snippets, OR add `# requires hardware` / `# requires GPU` markers for ones that don't.
3. **One page = one job.** A page either explains a concept, walks a tutorial, or lists API surface. Never all three.
4. **No emojis in code, no host paths.** `~/.strands_robots/` is fine; `/Users/cagatay/...` is not.
5. **Build must pass.** Run `mkdocs build --strict` after every page batch. A warning fails the build.
6. **Cross-link everything.** Each page ends with a "See also" block linking 2–3 sibling pages.

## Site map (matches mkdocs.yml)

```
- index.md                                 PAGE 01 — landing
- learning-path.md                         PAGE 02 — visual roadmap

- tutorial/
    - index.md                             PAGE 03 — tutorial index
    - 01-your-first-robot.md               PAGE 04
    - 02-simulation.md                     PAGE 05
    - 03-policies.md                       PAGE 06
    - 04-agents.md                         PAGE 07
    - 05-multi-robot.md                    PAGE 08
    - 06-recording.md                      PAGE 09
    - 07-training.md                       PAGE 10
    - 08-real-hardware.md                  PAGE 11
    - 09-advanced.md                       PAGE 12

- getting-started/
    - quickstart.md                        PAGE 13
    - installation.md                      PAGE 14
    - robot-factory.md                     PAGE 15

- robots/
    - index.md                             PAGE 16 — catalog (68 robots, 8 categories)
    - arms.md                              PAGE 17 — 22 arms
    - bimanual.md                          PAGE 18 — 3 bimanual
    - hands.md                             PAGE 19 — 8 hands
    - humanoids.md                         PAGE 20 — 18 humanoids
    - mobile.md                            PAGE 21 — 13 mobile + 2 aerial + 1 mobile_manip

- simulation/
    - overview.md                          PAGE 22
    - world-building.md                    PAGE 23
    - domain-randomization.md              PAGE 24
    - gymnasium-env.md                     PAGE 25 — describe: NOT YET IMPLEMENTED, point to roadmap

- policies/
    - overview.md                          PAGE 26 — Policy ABC, factory, registry
    - groot.md                             PAGE 27 — Gr00tPolicy + 4-tab matrix
    - lerobot-local.md                     PAGE 28 — LerobotLocalPolicy
    - custom-policies.md                   PAGE 29 — how to write one
    - gear-sonic.md                        PAGE 30 — describe as planned/external

- hardware/
    - robot-control.md                     PAGE 31 — strands_robots.hardware_robot
    - tools.md                             PAGE 32 — calibrate / camera / teleoperate / pose / serial

- training/
    - overview.md                          PAGE 33 — describe as planned/external (no trainer in main)

- recording.md                             PAGE 34 — DatasetRecorder + LeRobot v3

- examples/overview.md                     PAGE 35 — link to repo examples/

- architecture.md                          PAGE 36 — single source of truth diagram
- api-reference.md                         PAGE 37 — every public symbol
- contributing.md                          PAGE 38
- troubleshooting.md                       PAGE 39
```

## Per-page contract

Each page MUST contain:

1. Front-matter (only `description:` + `hide:` if no nav).
2. H1 with the page title.
3. A 1–2 sentence "what" intro.
4. A "TL;DR" code block when applicable (everything except policy of pages).
5. Body sections with H2/H3.
6. "Run it" tab when there is at least one runnable command.
7. "See also" footer.

## Build cadence per ambient cycle

Each cycle picks the next un-checked page below, opens the relevant code, writes the page, runs `mkdocs build --strict` against the *partial* site (pages not yet written are linked but missing — strict still fails on missing files; until a page exists, set its nav entry to a placeholder file via `touch docs/<path>`), then ticks the box.

When all 39 boxes are ticked, run a final `mkdocs build --strict --clean`, fix any warnings, then commit and push.

## Cycle checklist (work through top-down)

### Foundation (cycles 1-2)
- [x] **Cycle 1** — touch all 39 placeholder files so `mkdocs build` doesn't 404. Write index.md (page 01) AND learning-path.md (page 02). Build clean.
- [x] **Cycle 2** — write architecture.md (page 36). This is the single source of truth diagram every other page references.

### Tutorial track (cycles 3-7) — high impact, written first
- [x] **Cycle 3** — tutorial/index.md + 01-your-first-robot.md + 02-simulation.md
- [x] **Cycle 4** — tutorial/03-policies.md + 04-agents.md
- [x] **Cycle 5** — tutorial/05-multi-robot.md (uses mesh from PR101 if merged, else stub) + 06-recording.md
- [x] **Cycle 6** — tutorial/07-training.md + 08-real-hardware.md
- [x] **Cycle 7** — tutorial/09-advanced.md (covers Robot factory internals, custom backends, custom data_configs)

### Reference (cycles 8-12)
- [x] **Cycle 8** — getting-started/* (quickstart, installation, robot-factory)
- [x] **Cycle 9** — robots/index.md + robots/arms.md
- [x] **Cycle 10** — robots/{bimanual,hands,humanoids,mobile}.md
- [x] **Cycle 11** — simulation/{overview,world-building,domain-randomization,gymnasium-env}.md
- [x] **Cycle 12** — policies/{overview,groot,lerobot-local,custom-policies,gear-sonic}.md

### Hardware + ops (cycles 13-15)
- [x] **Cycle 13** — hardware/{robot-control,tools}.md
- [x] **Cycle 14** — recording.md + training/overview.md + examples/overview.md
- [x] **Cycle 15** — api-reference.md (auto-generated from docstrings, per-module)

### Polish (cycles 16-20)
- [ ] **Cycle 16** — contributing.md + troubleshooting.md. Verify every internal link resolves.
- [ ] **Cycle 17** — Walk every page, replace TODOs and stub code with real working examples. Add Run-It-Yourself tabs.
- [ ] **Cycle 18** — Add a "What's New" admonition at top of index pointing to recent merges (Robot factory, mesh, libero benchmark, etc.). Verify every robot in `registry/robots.json` is listed in the right category page with its render image.
- [ ] **Cycle 19** — `mkdocs build --strict --clean`. Open every warning. Fix every broken link, every missing image, every orphan page. Run `mkdocs serve` once and visually walk the nav tree.
- [ ] **Cycle 20** — Final pass: write the PR description, push to fork, open PR. End with `[AMBIENT_DONE]`.

## Code-grounding cheatsheet (open these files when writing)

| Page topic | Files to read first |
|------------|---------------------|
| Robot factory | `strands_robots/robot.py`, `strands_robots/registry/robots.py` |
| Hardware Robot | `strands_robots/hardware_robot.py` |
| Simulation | `strands_robots/simulation/__init__.py`, `simulation/factory.py`, `simulation/mujoco/simulation.py` |
| Policies | `strands_robots/policies/{base,factory,mock}.py`, `policies/groot/policy.py`, `policies/lerobot_local/policy.py` |
| Tools | `strands_robots/tools/*.py` |
| Recording | `strands_robots/dataset_recorder.py` |
| Robot list | `strands_robots/registry/robots.json` (68 robots) |
| Examples | `examples/` directory at repo root |

## Done definition

- [ ] `mkdocs build --strict --clean` exits 0 with zero warnings
- [ ] Every page in the site map has substantive content (>30 non-blank lines)
- [ ] Every code block compiles (smoke-tested)
- [ ] Every robot in `registry/robots.json` appears on its category page
- [ ] PR opened against `strands-labs/robots:main` from `cagataycali:docs/mkdocs-site`
- [ ] Final response on the autonomous loop ends with `[AMBIENT_DONE]`
