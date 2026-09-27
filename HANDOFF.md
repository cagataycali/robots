# Lane train - HANDOFF

Branch `dash/lane-train`, worktree `/Users/cagatay/robots-dash/train`. Three commits, my paths only.

## Files added (strands_robots/dashboard/)

| file | LOC | from | adapted how |
|---|---|---|---|
| `routes_train.py` | 420 | branch server.py 638, 811-958, 1561, 1675-1797 | one `APIRouter(prefix="/api", tags=["train"])`, every route `Depends(access.require_session)`, state via `request.app.state.<x>` with `getattr` for other lanes' state (`bridge`, `devices`, `record`) |
| `training.py` | 690 | branch verbatim | record-lane imports (`dataset_check`, `output_dir_check`) made lazy; added path homes + `PathOutside` + `contain*()`; added `spec_problems()` field grader; `_NOT_IN_FORM` archaeology ("Q37") dropped |
| `checkpoints.py` | 470 | branch verbatim | fallback-family archaeology dropped; `local_checkpoints` reads `_hf_cache_root()` (was a hardcoded path beside the helper that exists for it); docstring on `declared_features` |
| `deploy.py` | 240 | branch verbatim | `camera_liveness` (devices lane) imported lazily inside `render_snippet` |
| `policy_fit.py` | 270 | branch verbatim + `config_api._policy_catalog` | `policy_catalog()`, `WIRE_CMD_KEYS`, `WIRE_KEY_TYPES` ported here because no lane ports `config_api.py` and `/api/policies` needs them |
| `artifact_check.py` | 160 | branch verbatim | none |
| `calibration.py` | 260 | branch + smallest helpers of the deleted `tools/lerobot_calibrate.py` | `default_root()` (lerobot constants -> env -> default), `structure()`, `calibration_info()`, `listing()` rendering the exact markdown `frontend/src/lib/calibration.ts` parses |
| `calibration_run.py` | 330 | branch verbatim | em dashes -> `-`; `_calibrate_argv()` refuses through `utils.require_optional(..., extra="lerobot")`; `CONFIRM_KEY` |

`attach(app)` appends `_close_calibration_runs` to `app.state.shutdown_hooks` (creates the list if absent).

## Routes served (20) and how each was verified

Smoke = `FastAPI(); include_router(router); attach(app)` + `TestClient`, run twice: open posture
(TestClient's `testclient`/`testserver` pair is admitted by `access.py` by design) and closed
posture (`STRANDS_DASH_AUTH_ENABLED=1 DASHBOARD_AUTH_TOKEN=...`, bearer header).

| route | result |
|---|---|
| GET /api/training/trainers | 200 `{trainers[8], unsupported{cosmos3,fast_sac,fast_td3,ppo,sagemaker}, fields, output_home}` |
| GET /api/training/datasets | needs record lane's `dataset_check.mark_live_recording` (imported by name at call time); not smoked in this worktree |
| GET /api/training/jobs | 200 `{jobs: [], problem: null}` |
| POST /api/training/validate | bad spec -> **422** `{message, fields:{step: unknown field..., steps: ...positive integer..., output_dir: must be inside <home>}}`; clean spec -> trainer's own validate() result |
| GET /api/training/output-dir | empty -> 422; `/tmp/train_x` -> **400** `output_dir must be inside ~/.strands_robots/training`; inside -> `inspect_output_dir` verdict (record lane's `output_dir_check`) |
| POST /api/training/submit | graded like validate, `confirm_clear` passthrough, `bridge.record_activity` when a bridge is mounted |
| GET /api/training/status | 200 (mock provider: `[mock] job j1: success`) |
| POST /api/training/export | output_dir outside home -> 400; dataset_root contained too |
| GET /api/checkpoints/search?q=x | 200 (Hub answered: xvla rows; `hf_auth` present) |
| GET /api/checkpoints/features | `/etc/passwd` -> 400 (homes: training output + HF hub cache); `lerobot/nothing` -> `{}` |
| GET /api/checkpoints/families | 200, 14 families from lerobot's registry |
| GET /api/policies | 200, 14 providers with `wire_fields` / `wire_safe` / `server_based` |
| POST /api/policies/validate | 200 `{ok:false, stage:"trust", error:..., needs_consent}` for lerobot_local (trust gate), consent attached from the exception's code |
| POST /api/deploy/snippet | 200, script rendered; `serial` without a device manager -> 404 |
| GET /api/calibration | 200 `{status, text (markdown), root, count, entries}` |
| GET /api/calibration/{name} | 404 with hint when absent; 409 with candidates when ambiguous |
| POST /api/calibration/run | no session -> **401**; session but no `confirm: true` and no grant -> **403** with `code: calibration_confirm_required` + `needs_consent` card; `confirm: true` + bad role -> 422; under the grant -> 200 run started (real pty, `step: starting`), then status 200, cancel 200 `alive: false` |
| GET /api/calibration/run/{sid} | 404 for unknown sid |
| POST /api/calibration/run/{sid}/key | 409 on a finished run / bad key |
| POST /api/calibration/run/{sid}/cancel | SIGTERM the process group, SIGKILL after 3 s |

`ruff check` + `ruff format` clean on all eight files; `python -c "import strands_robots.dashboard.routes_train"` clean; no em/en dashes.

## Decisions the coordinator should know

1. **Path homes.** `output_dir` must resolve inside `STRANDS_TRAIN_OUTPUT_DIR` or `<base_dir_path()>/training`
   (`~/.strands_robots/training`). `dataset_root` must resolve inside `HF_LEROBOT_HOME` (default
   `~/.cache/huggingface/lerobot`), any `STRANDS_ROBOTS_DATA_DIRS` entry, or the parent of a root the collect
   wizard remembered. A checkpoint *path* (not an `org/name` id) must be inside the training home or the HF hub
   cache. Refusal is 400 with the same sentence whether or not the path exists. **Frontend impact:** the branch's
   `lib/outputDirSuggest.ts` suggests `/tmp/train_<name>`, which is now refused; it should suggest
   `${output_home}/train_<name>` using the new `output_home` key on GET /api/training/trainers.
2. **422 field errors.** The branch answered a bad spec with 200 `{status:"error", text}`. Now unknown fields,
   non-count numbers, a leading `-`, an unknown trainer, a missing data source/output_dir and an out-of-home path
   are 422 `{message, fields:{<field>: <why>}}` (nested under `error` by main's exception handler). A spec that
   passes the grader still goes through `train_policy(action="validate")`, whose result keeps the branch shape.
3. **Calibration consent.** `consent.KINDS` is closed and has no calibration kind, so the refusal carries the
   `agent_physical_motion` card (its `message` is the calibration refusal, its `title`/`risk` are the kind's).
   Two ways through: `confirm: true` in the POST body (the wizard's own confirm sheet, which the branch's
   `CalibrateWizard.tsx` already shows before posting - it needs to add the field) or the standing
   `STRANDS_DASH_AGENT_PHYSICAL_MOTION=1` grant. A dedicated `calibration` consent kind in `consent.py` is a
   coordinator decision; the route would then pass a `{code: ...}` refusal instead of the motion verdict.
4. **`/api/policies` catalog** lives in `policy_fit.policy_catalog()` (was `config_api._policy_catalog`). The
   mesh lane's `/api/robots/{pid}/task` may want `WIRE_CMD_KEYS` from the same place.
5. **lerobot is optional** everywhere here: listing calibrations never imports it; the wizard refuses with
   `require_optional(..., extra="lerobot")` -> 501 `{error, extra:"lerobot"}`; `policy_families()` and the
   family matcher fall back to hand lists.

## Deps
Nothing new. `huggingface_hub` is used lazily (search degrades to local-only with a `problem` sentence).

## Gaps
- GET /api/training/datasets and GET /api/training/output-dir (inside-home path) exercise record-lane modules
  (`dataset_check`, `output_dir_check`) that are not in this worktree; both import by the branch's public names
  at call time and were not smoked here.
- `/api/deploy/snippet` with `serial` needs devices-lane `app.state.devices.profiles`; `/api/calibration/run`
  uses `app.state.devices.port_owner` when present (skips the bus-collision check when the device manager is not
  mounted, still gated by confirm/grant).
- No integration of `record_activity` for export/calibration-cancel; only submit and calibration start, as the
  branch did (plus calibration start, which the branch did not record).

## Needs coordinator
- `server.create_app()`: `app.include_router(routes_train.router)`; `routes_train.attach(app)`; run
  `app.state.shutdown_hooks` in `_lifespan`.
- Frontend lane: `outputDirSuggest.ts` -> `output_home`; `CalibrateWizard.tsx` -> send `confirm: true` after its
  sheet, and render `needs_consent` on the 403; `TrainingTab.tsx` -> render 422 `fields` per field.
- Decide on a `calibration` consent kind (see 3).
- `docs/` + changelog fragment for the new routes and the two env variables (`STRANDS_TRAIN_OUTPUT_DIR` is new;
  `STRANDS_ROBOTS_DATA_DIRS` and `HF_LEROBOT_HOME` were already read by the branch).
