# HANDOFF - lane frontend (branch `dash/lane-frontend`)

Owner paths: `strands_robots/dashboard/frontend/**` (source), `strands_robots/dashboard/static/**` (built output).
Commits: `8063781` (SPA restored, /ws/agent), `13feb5b` (Sim tab, e-stop rails, /api/config fallback, build into static/).

## What is here

The React/Vite operator SPA from `fork/dashboard-revamp` (59fc905d) is the dashboard UI on `main`:
built, committed, served by `server.py` as it stands. No node at runtime; `npm run build` is a
developer step whose output (`static/index.html`, `static/assets/index.{js,css}`) is committed.

    static/                      1.3 MB total (committed)
      index.html                 1.4 KB   the SPA shell; asset URLs are /static/assets/...
      assets/index.js            507 KB   (163 KB gzip) React + the whole app, one chunk, fixed name
      assets/index.css            68 KB
      twin.js                    8.5 KB   the three.js twin (main's file, +1 optional ctor arg)
      vendor/three.module.min.js 692 KB   untouched
      vendor/OrbitControls.js     32 KB   untouched
      icon.svg, apple-touch-icon.png

`vite.config.ts`: `base: '/static/'`, `outDir: '../static'`, `emptyOutDir: false`, fixed asset
names (`assets/index.js`, not hashed) so a rebuild overwrites instead of accumulating siblings in a
directory the build does not empty. **server.py needs no change**: it already mounts `/static` and
serves `static/index.html` at `/`.

### Removed vs the branch
- 115 `src/lib/*.test.mjs` files and `scripts/run-lib-tests.mjs` (owner decision: tests land later).
  `scripts/gen-bundle-routes.mjs` is kept (`npm run gen:routes` regenerates `bundleRoutes.generated.ts`,
  the list the "older server" banner compares against `/openapi.json`; regenerated in this lane: 79 routes).
- `vite-plugin-pwa` and the service worker. Under `/static/` a worker could not control `/`, and a
  cached shell / replayed request is a liability on a page that moves motors. `usePwa` keeps
  install prompt, online state and wake lock; `needRefresh` is always false, `update` is a reload.
  `AuthGate` lost its SW hook. Manifest icons (192/512/maskable) dropped with it.
- `static/index.html`, `static/app.js`, `static/app.css` (main's vanilla UI) - replaced by the SPA.
- The `?token=` query on WebSocket URLs (`endpoints.wsUrl`). main's `access.presented_token`
  reads Bearer or the `strands_dash` cookie and never the query string; the login ceremony sets
  the cookie, so sockets are admitted exactly when the page is. See gap 1 below.

### Adapted to main's rails
| surface | branch | now |
|---|---|---|
| agent | `/ws/chat` frames `chat`/`interrupt_response`, `/api/agent/status`, `POST /api/agent/reset` | `/ws/agent` (`routes_agent.py`): send `{type:'say',text}`, `{type:'resume',id,approve,always}`; receive `text`/`tool_use`/`tool_result`/`interrupt`/`done`/`error`. `GET /api/agent` fills the model chip. "clear" closes the socket (one Console per socket). Consent card renders `MotionGate` reasons `{tool,session_id,positions,detail}` plus the two branch shapes; new "yes, and stop asking for this one" button = `always:true`. |
| e-stop | `POST /api/safety/estop` (mesh, per-peer answers) | `EstopSheet` takes `meshBacked` (= `/ws/mesh` says `mesh.online`). Mesh-backed: `POST /api/mesh/safety/estop` + `/api/mesh/safety/resume {override_code}` (branch payloads) AND the sim rail. Sim-only: `POST /api/safety/estop` -> `{lockout, frozen}`, `POST /api/safety/resume` -> `{lockout, thawed}`, rendered from `lockout.as_fields()`. A stop always fires the local sim rail first: it is never refused. |
| settings | `GET/POST /api/config` (composite ConfigDoc) | `lib/configDoc.ts`: tries `/api/config`; on 404 composes the ConfigDoc from `GET /api/settings` (`{settings:{section:{key}}, file}`) + `GET /api/agent`, and saves via `POST /api/settings` (sectioned patch; `{changed, errors}` -> ApplyResult, non-schema keys reported as `ignored`). Policies/env rows/mesh info are empty in fallback mode; the drawer renders empty lists. |
| socket refusals | close code 1008 | 1008 **or 4401** (`access.refuse_socket`) in chatDelivery, cameraRetry, voiceSession. |
| auth | `/api/auth/*` | unchanged: same paths and payload keys (`challenge_id`, `options`, `credential`, `token`, `label`, `bootstrap`; status `enabled/setup_required/bootstrap_required/rp_id/secure_context/rpid_usable/authenticated`). `PasskeyList` reads `/api/auth/credentials` (main serves it). |

### New: Sim tab (`components/SimTab.tsx`, `lib/twinLoader.ts`, `static/twin.js`)
Nav chip "sim", `?panel=sim`. Speaks `routes_sim.py` verbatim:
`GET/POST /api/sim`, `GET /api/sim/ports`, `GET/DELETE /api/sim/{sid}`, `POST /api/sim/{sid}/joints {positions:{name:rad}}`
(sliders coalesced to one POST / 80 ms), `POST /api/sim/{sid}/reset`, `GET /api/sim/{sid}/stream.mjpg` (only while the
Camera view is shown), `WS /ws/telemetry/{sid}?poses=1` (JSON snapshot drives the joint strip and the lockout line;
the binary frame goes to `Twin.poses`), `GET /api/safety` + `/estop` + `/resume` for the in-tab lockout line/button.
Robot list: `GET /api/fleet?mode=sim` (main's fleet.py rows with `has_sim`/`model_local`); if that route answers
peers instead (mesh lane), falls back to `GET /api/robots/registry` filtered by `has_sim`.
The twin is **not bundled**: `twinLoader.ts` does `import('/static/twin.js')` at runtime (`@vite-ignore`), so three.js
is fetched only by a page that opens the tab and the shipped file is the one that runs. `twin.js` gained an optional
third ctor arg `{base, headers}` so the React page can pass its backend base and bearer token; default behaviour
(same origin, cookie) is unchanged, and `_get()` now throws on a non-2xx instead of parsing an error body.

## Verified (how)
- `npm ci && npm run build` green (tsc -b strict + vite; 188 modules, 0.5 s).
- Server from this worktree: `PYTHONPATH=$PWD /Users/cagatay/robots/.venv/bin/python -m uvicorn --factory strands_robots.dashboard.server:create_app --port 8791`.
  `curl /` -> 200 text/html, the SPA index; `/static/assets/index.js|css`, `/static/twin.js`, `/static/vendor/*`, `/static/icon.svg` all 200.
- Headless Chrome (puppeteer-core + system Chrome, SwiftShader GL) against that server, loopback open posture:
  - page boots with **0 pageerrors**; only 404s are `/api/config` (falls back), `/ws/mesh`, `/api/record/session` (lanes not merged yet - the "older server" banner lists them, as designed).
  - Sim tab: started `so101` (physics), telemetry at ~10.8 fps, twin loaded scene `ngeom 31 / 13 meshes / pose_row_floats 12` and rendered the arm (screenshot checked), Camera view streams 512x384 MJPG, slider on joint 1 -> `qpos 0.598`, E-STOP (sim) -> session `FROZEN` + lockout `locked`, RESUME -> `unknown`, stop -> 0 sessions.
  - Agent dock over `/ws/agent`: "which sim sessions are running?" -> `tool_use sim_sessions` chip + streamed answer; model chip from `/api/agent`; "1 tools ask first".
  - Settings drawer renders via the `/api/settings` fallback (Connection / Agent / Voice / Mesh / Env / Security tabs).

## Needs coordinator
1. **`/api/config`** (branch server.py 1234-1246, `config_api.snapshot/apply`) is owned by no lane. Port it (or a thin
   adapter over `settings.py` + `redacted_settings` + the policy catalog + `env_view`) to light up the Env tab, the
   policy catalog in the run form, and mesh info in Settings. Until then the SPA runs in fallback mode (above) and
   `POST /api/settings` writes `agent/voice/mesh/runtime/security`; `reset_prompt`, `reset_agent`, `env` patches are
   reported as "not recognised, so not saved".
2. **`/api/fleet` collision**: main's `fleet.py` (registry rows + peers) and the mesh lane's port of the branch's
   `/api/fleet` (bridge snapshot) share one path. The SPA reads the fleet from `/ws/mesh`; the Sim tab tolerates either
   shape (see above). Decide which one keeps the path.
3. **`websockets`** was missing from `~/robots/.venv` (uvicorn logged "No supported WebSocket library detected" and every
   `/ws/*` handshake was 404). It is a core dep in pyproject (`websockets>=17.0`), so a normal install is fine; I
   `uv pip install`ed it into the venv. Consider `uvicorn[standard]` in the `dashboard` extra so a venv built from
   the extra alone cannot hit this.
4. **Cross-origin backend + WebSockets**: the drawer still lets a page point at another host (`?backend=`). Fetches carry
   Bearer, but sockets have no header and the cookie is first-party only, and main refuses `?token=`. If that workflow
   matters, the server should accept the token as a `Sec-WebSocket-Protocol` entry (the page can then send it). Not
   done here: server-side.
5. `GET /api/sim/{sid}/stream.mjpg` is an `<img src>`: cookie-authenticated on the same origin (works), Bearer-only
   sessions would need the same server-side decision as 4. The branch's `AuthedImg` is kept for still previews.
6. `docs/`: `server.py`'s module docstring still says "The static UI is plain files ... no build step"; update it
   (coordinator-owned) to: built SPA committed under `static/`, rebuild with `cd strands_robots/dashboard/frontend && npm ci && npm run build`.
7. `/ws/voice` (`useVoice`/`voiceSession.ts`) keeps the branch protocol; coordinator's `voice.py` must match it.

## Developer notes
- `npm run dev` proxies `/api`, `/ws`, `/static/twin.js`, `/static/vendor` to `localhost:8080`.
- Rebuild = `npm run build` in `frontend/`; commit `static/index.html` + `static/assets/*` with the source change.
- `npm run gen:routes` after adding or renaming an `/api/...` literal in `src/`.
- `frontend/.gitignore` ignores `node_modules`; `dist` is repo-ignored and unused (output goes to `../static`).
