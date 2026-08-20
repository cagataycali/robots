/**
 * A directory an abandoned recording left behind cannot be trained on or replayed by accident (Q37).
 *
 * The verdict is unit-tested on the server and the lib is asserted in node — the same position Q35
 * and Q36 were in while the screen showed nothing. What only a browser can answer: is the row
 * MARKED where the choice is made, is replay actually dead, and does pressing train send NOTHING?
 *
 * The submit assertion is the one that matters and it is made by COUNTING REQUESTS, not by reading
 * a toast: "training did not start" is a claim about the network, and a message saying so is not
 * evidence. /api/training/submit is intercepted and counted (and answers an error if it is ever
 * reached), so a gate that leaks shows up as a count, not as prose.
 *
 * Injected: datasets (one abandoned + one healthy local + one Hub row), jobs (empty), submit.
 * Nothing trains, no real dataset is touched, no robot is addressed.
 *
 * Run: node scripts/audit-dataset-abandoned-hold.mjs
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const ABANDONED_PROBLEM =
  '0 episodes. meta/info.json is written when a recording session OPENS, before the first episode is '
  + 'captured, so a directory like this is what an abandoned session leaves behind - not a dataset. '
  + 'Record into it, or delete it.'

const DATASETS = {
  datasets: [
    { root: '/tmp/audit/local/sim_recording', repo_id: 'local/sim_recording', total_episodes: 0, fps: 30,
      robot_type: 'so101', usable: false, reason: 'no_episodes', problem: ABANDONED_PROBLEM },
    { root: '/tmp/audit/cagataydev/good-one', repo_id: 'cagataydev/good-one', total_episodes: 30, fps: 10,
      robot_type: 'so101', usable: true, note: 'read from meta/info.json only' },
    { repo_id: 'lerobot/pusht', local: false, downloads: 91234, total_episodes: 206, fps: 10 },
  ],
}

const browser = await chromium.launch()
// serviceWorkers:'block' is REQUIRED: a PWA-cached response is not interceptable, so the fixtures
// would silently never land and every assertion below would be about the real fleet.
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))

let submits = 0
await page.route('**/api/training/datasets**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(DATASETS) }))
await page.route('**/api/training/jobs', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ jobs: [] }) }))
await page.route('**/api/training/submit', r => { submits += 1; r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'error', text: 'blocked by audit - nothing was started' }) }) })
// A replay click would spawn a real sim peer. Counted and refused for the same reason.
let replays = 0
await page.route('**/api/replay**', r => { replays += 1; r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'error', text: 'blocked by audit' }) }) })

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.locator('button.chip:has-text("train")').first().click()
const picker = page.locator('select').filter({ has: page.locator('option:has-text("pick a dataset")') }).first()
await picker.waitFor({ timeout: 20000 })
// Wait for the fixture rows THEMSELVES, never for a duration: the list arrives from a debounced
// search and the previous audit in this family lost 20 minutes to exactly that mistake.
// state:'attached' — an <option> is never "visible" to playwright (its box lives inside the
// closed select), so the default visibility wait times out on an element that is right there.
await page.locator('option:has-text("sim_recording")').first().waitFor({ state: 'attached', timeout: 20000 })

// ---- 1. the mark is where the choice is made
const opts = await page.locator('select option').allInnerTexts()
const abandonedOpt = opts.find(o => o.includes('sim_recording')) ?? ''
const goodOpt = opts.find(o => o.includes('good-one')) ?? ''
if (!abandonedOpt.includes('⚠')) failures.push(`the abandoned dataset's option is not marked: "${abandonedOpt.trim()}"`)
if (goodOpt.includes('⚠')) failures.push(`a healthy dataset is marked as a problem: "${goodOpt.trim()}"`)

// ---- 2. replay is dead on it, alive on the healthy one, with the reason readable
const rowBtn = (repo) => page.locator('.train-job', { hasText: repo }).locator('button:has-text("replay")').first()
if (await rowBtn('sim_recording').isEnabled()) failures.push('replay is clickable on a dataset with no episode 0')
const title = await rowBtn('sim_recording').getAttribute('title')
if (!/OPENS/.test(title ?? '')) failures.push(`the dead replay button does not explain itself: "${title}"`)
if (!(await rowBtn('good-one').isEnabled())) failures.push('replay went dead on a healthy dataset')

// ---- 3. pressing train sends NOTHING
await picker.selectOption('/tmp/audit/local/sim_recording')
const out = page.locator('input[placeholder*="output"], input[name="output_dir"]').first()
if (await out.count()) { await out.fill('/tmp/audit-out') }
else {
  // The form's output_dir field is required for the train button to enable; find it by label.
  const byLabel = page.locator('label:has-text("output") input').first()
  if (await byLabel.count()) await byLabel.fill('/tmp/audit-out')
  else failures.push('could not find the output_dir field - the train button may be disabled for the wrong reason')
}
const train = page.locator('button:has-text("train")').filter({ hasNot: page.locator('.chip') }).last()
await train.click()
await page.locator('.artifact-hold').first().waitFor({ timeout: 8000 }).catch(() => {})
const hold = page.locator('.artifact-hold').first()
if (!(await hold.count())) failures.push('pressing train on an abandoned dataset produced no explanation')
else {
  const text = await hold.innerText()
  if (!text.includes('OPENS')) failures.push('the refusal does not name the mechanism')
  if (!/train on it anyway/i.test(text)) failures.push('the refusal offers no door')
  if (!/pick another dataset/i.test(text)) failures.push('the refusal offers no way to decline')
}
if (submits !== 0) failures.push(`the gate leaked: ${submits} submit request(s) reached the server`)

// ---- 4. the door opens exactly once, for exactly this dataset
await page.locator('.artifact-hold button:has-text("train on it anyway")').first().click()
await train.click()
await page.waitForTimeout(1500)
if (submits !== 1) failures.push(`"train on it anyway" did not let the run through (submits=${submits})`)

// ...and insisting on THIS dataset must not silence the check for the next one.
await picker.selectOption('/tmp/audit/cagataydev/good-one')
await picker.selectOption('/tmp/audit/local/sim_recording')
await train.click()
await page.locator('.artifact-hold').first().waitFor({ timeout: 8000 }).catch(() => {})
if (!(await page.locator('.artifact-hold').count())) failures.push('the override survived a dataset change - the check is now permanently off')
if (submits !== 1) failures.push(`a second run slipped through after re-picking (submits=${submits})`)

// ---- 5. a healthy dataset starts with no interruption at all
await picker.selectOption('/tmp/audit/cagataydev/good-one')
await train.click()
await page.waitForTimeout(1500)
if (await page.locator('.artifact-hold').count()) failures.push('a healthy dataset was held back')
if (submits !== 2) failures.push(`a healthy dataset did not start (submits=${submits}, expected 2)`)

if (replays !== 0) failures.push(`a replay was started by this audit (${replays}) - it should never have been reachable`)
if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)

await browser.close()
if (failures.length) { console.log('FAILURES:'); for (const f of failures) console.log(`  ✗ ${f}`); process.exit(1) }
console.log('dataset hold: marked in the picker, replay dead with the reason, train sends NOTHING, the door opens once per dataset, healthy datasets untouched')
