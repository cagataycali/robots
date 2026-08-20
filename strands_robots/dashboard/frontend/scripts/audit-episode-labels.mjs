/**
 * The #2486 labels panel ON THE PAGE: the verdicts render, the impossible cases explain themselves,
 * and a route the running server lacks says "restart", not "HTTP 404".
 *
 * Why an audit and not just the lib tests: episodeLabels.ts and serverSkew.ts are proven pure, but
 * this dashboard has burned a whole iteration before on a rule that was CORRECT in lib/ and inert in
 * the UI (the fleet badge rendered from the websocket snapshot, not the route that was changed). The
 * three claims below can only fail in the wiring — a gate that never reaches `disabled`, a summary
 * whose sentence is swallowed, an error branch that keeps the raw HTTP text.
 *
 * The labels API landed after the running dashboard booted (c3a835f9, and loops never restart it —
 * camera TCC law), so the route is fulfilled here; the backend half is proven by
 * tests/test_dashboard_episode_labels.py.
 *
 * READ-ONLY: opens the train screen and toggles one disclosure. Nothing spawns, records or moves.
 *
 * Run: node scripts/audit-episode-labels.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(process.env.HOME + '/.strands_dashboard/local_api_token.txt', 'utf8').trim()
const DATASETS = {
  datasets: [
    { repo_id: 'me/pick-cube', root: '/tmp/ds/pick-cube', label: 'pick-cube', total_episodes: 3, fps: 30 },
    { repo_id: 'me/being-recorded', root: '/tmp/ds/being-recorded', total_episodes: 0, recording: true },
    { repo_id: 'hub/someone-elses', downloads: 1234 },  // no root: lives on the Hub only
  ],
}
const LABELS = {
  benchmark: 'cube_lift', schema_version: 1, total_episodes: 3, with_verdict: 2, labelled: 1, disputed: 1,
  can_annotate: true, why: '1 episode(s) carry a verdict and are waiting for a quality grade',
  episodes: [
    { episode_index: 0, verdict: 'success', steps: 150, quality: 'high', failure_mode: 'near_miss',
      note: 'clean lift', disputes_verdict: true, model: 'human', annotatable: true },
    { episode_index: 1, verdict: 'failure', steps: 42, quality: null, failure_mode: null, note: null,
      disputes_verdict: false, model: null, annotatable: true },
    { episode_index: 2, verdict: null, quality: null, failure_mode: null, note: null,
      disputes_verdict: false, model: null, annotatable: false },
  ],
}
const NO_SIDECAR = {
  episodes: [], with_verdict: 0, labelled: 0, disputed: 0, can_annotate: false,
  why: 'no episode_labels.json in this dataset: a real-arm recording has no predicate verdict to annotate — this is a gap in the label rail, not a permission problem',
}

const failures = []
const b = await chromium.launch()
const ctx = await b.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const p = await ctx.newPage()
p.on('pageerror', e => failures.push(`PAGEERR ${e.message.slice(0, 160)}`))
await p.routeWebSocket('**/ws/mesh**', ws => ws.send(JSON.stringify({ type: 'snapshot', dashboard_peer_id: 'a', peers: {} })))
await p.route('**/api/training/datasets**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(DATASETS) }))

let labelsAnswer = { status: 200, body: LABELS }
await p.route('**/api/datasets/labels**', r => r.fulfill({
  status: labelsAnswer.status, contentType: 'application/json', body: JSON.stringify(labelsAnswer.body),
}))

await p.goto('http://127.0.0.1:8090/?token=' + TOKEN, { waitUntil: 'domcontentloaded' })
await p.waitForTimeout(4500)
await p.locator('button.chip:has-text("train")').first().click()
await p.waitForTimeout(2500)

const rowOf = (repo) => p.locator('.train-job', { has: p.locator('b', { hasText: repo }) }).first()
const labelBtn = (repo) => rowOf(repo).locator('button:has-text("labels")')

// CLAIM 1 — the gate reaches the DOM: no button where the panel cannot work, and the reason is on it.
for (const [repo, expect] of [['hub/someone-elses', /sidecar next to the dataset on disk/],
                              ['me/being-recorded', /judged after the session/]]) {
  const btn = labelBtn(repo)
  if (!await btn.count()) { failures.push(`${repo}: no labels button at all`); continue }
  if (!await btn.isDisabled()) failures.push(`${repo}: labels button is clickable — it would 404 or show nothing`)
  const title = (await btn.getAttribute('title')) || ''
  if (!expect.test(title)) failures.push(`${repo}: reason missing from the button — got "${title.slice(0, 80)}"`)
}
const local = labelBtn('me/pick-cube')
if (!await local.count()) failures.push('the local dataset has no labels button')
else if (await local.isDisabled()) failures.push('the local dataset row refuses its own labels — over-refusal hides a working feature')

// CLAIM 2 — clicking renders the verdicts, and the unjudged/unverdicted episodes say what is missing.
await local.click()
await p.waitForTimeout(800)
const panel = rowOf('me/pick-cube').locator('.ds-labels')
if (!await panel.count()) failures.push('the labels panel did not open')
else {
  const text = await panel.innerText()
  for (const want of [/1\/2 judged/, /benchmark cube_lift/, /1 disputing the verdict/,
                      /quality high/, /near_miss/, /judge disputes this verdict/, /by human/,
                      /awaiting a quality grade/, /cannot be annotated/]) {
    if (!want.test(text)) failures.push(`panel text is missing ${want}: ${text.replace(/\s+/g, ' ').slice(0, 200)}`)
  }
  // The unverdicted episode must not wear a verdict-shaped badge that reads as a pass.
  const muted = await panel.locator('.ds-label-row.muted').count()
  if (muted < 2) failures.push(`expected the ungraded and unverdicted rows to render muted, got ${muted}`)
}

// CLAIM 3 — the impossible case shows its SENTENCE, never a bare "0 labelled".
labelsAnswer = { status: 200, body: NO_SIDECAR }
await local.click(); await p.waitForTimeout(400); await local.click(); await p.waitForTimeout(900)
{
  const text = (await rowOf('me/pick-cube').locator('.ds-labels').innerText()).replace(/\s+/g, ' ')
  if (!/not a permission problem/.test(text)) failures.push(`the no-sidecar explanation did not render: ${text.slice(0, 200)}`)
  if (/0\/0 judged/.test(text)) failures.push('an impossibility rendered as a count — the exact confusion #2486 removed')
}

// CLAIM 4 — a route the running server does not have says RESTART, not HTTP 404. The explanation
// comes from the FETCH LAYER (lib/serverAge via api()), so this also proves that one mechanism
// reaches a screen that does nothing special — iteration 217 deleted the duplicate that only this
// panel had.
labelsAnswer = { status: 404, body: { error: 'not found', detail: 'no endpoint at /api/datasets/labels' } }
await local.click(); await p.waitForTimeout(400); await local.click(); await p.waitForTimeout(900)
{
  const text = (await rowOf('me/pick-cube').locator('.ds-labels').innerText()).replace(/\s+/g, ' ')
  if (!/does not have \/api\/datasets\/labels/i.test(text)) failures.push(`staleness is not explained on the page: ${text.slice(0, 200)}`)
  if (!/running code from before this feature existed/i.test(text)) failures.push('the page does not say the server is old')
  if (!/restart the dashboard from a terminal/i.test(text)) failures.push('the page does not name the action that fixes it')
  if (!/no camera access/i.test(text)) failures.push('the page does not warn that a daemon restart comes back blind')
  if (/^HTTP 404$/.test(text.trim())) failures.push('the raw HTTP text survived to the screen')
}

await b.close()
if (failures.length) { console.log('FAIL\n' + failures.map(f => '  - ' + f).join('\n')); process.exit(1) }
console.log('PASS  episode labels: gate, verdicts, the unlabellable explanation, and version skew all render')
