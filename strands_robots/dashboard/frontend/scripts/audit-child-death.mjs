/**
 * A dead managed child names its cause on the screen — U22, in a browser.
 *
 * The unit tests pin the sentences; only a browser can answer whether they reach the
 * row, and whether the LOG TAIL under them is captioned honestly. The live defect this
 * guards: the sim twin was SIGKILLed 22h after the last line in its ring buffer, and
 * the drawer rendered "· exited" over that startup burst — a robot that warned and
 * stopped itself. Nothing in the payload had changed; the page simply never read
 * `returncode`.
 *
 * /api/devices is INJECTED, so this audit starts no process and touches no arm. The
 * fixture carries four children on purpose: a kill, a crash, a clean exit and one still
 * alive — the point is that they no longer share a sentence.
 *
 * Run: node scripts/audit-child-death.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import { blockMutations } from './lib/audit-guard.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

/* The clock in a log line is the SERVER's local wall clock, and started_at is an epoch
   second, so the fixture builds both from one Date — otherwise this audit would pass or
   fail depending on the machine's timezone, which is the very trap childDeath.ts avoids. */
const startedAt = Math.floor(Date.now() / 1000) - 22 * 3600
const at = (offsetS) => {
  const d = new Date((startedAt + offsetS) * 1000)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

const doc = {
  serial_ports: [], cameras: [], camera_names: [], camera_problem: null,
  managed: {
    // The real corpse, reproduced: killed, and its ring holds only its first seconds.
    'twin': {
      peer_id: 'twin', robot_name: 'so101', mode: 'sim', alive: false, returncode: -9,
      started_at: startedAt,
      log_tail: [`${at(0)} [safety:twin] No emergency-stop resume code set.`,
                 `${at(1)} [safety:twin]   To allow remote resume set STRANDS_MESH_OVERRIDE_CODE`],
    },
    // A Python failure, and a ring that really is its last words (spread over hours).
    'crashed-arm': {
      peer_id: 'crashed-arm', robot_name: 'so101', mode: 'real', alive: false, returncode: 1,
      started_at: startedAt,
      log_tail: [`${at(0)} hardware connected`, `${at(5 * 3600 + 137)} ConnectionError: Port is in use!`],
    },
    'finished-job': {
      peer_id: 'finished-job', robot_name: 'so101', mode: 'collect', alive: false, returncode: 0,
      started_at: startedAt, log_tail: [`${at(0)} recording 5 episodes`],
    },
    'live-arm': {
      peer_id: 'live-arm', robot_name: 'so101', mode: 'real', alive: true, returncode: null,
      started_at: startedAt, log_tail: [`${at(0)} hardware connected`],
    },
  },
}

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
/* The audit hardware guard goes FIRST: playwright matches handlers in REVERSE registration order, so
   every fixture below still wins, and any MUTATING request this audit forgot to intercept is blocked
   and recorded instead of reaching the running dashboard (which spawns processes and commands arms). */
const guard = await blockMutations(page)
await page.route('**/api/devices', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(doc) }))
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.locator('button.chip:has-text("devices")').first().click()
await page.waitForSelector('li:has-text("twin")', { timeout: 15000 })

const rowText = async (peer) => (await page.locator(`li:has-text("${peer}")`).first().innerText()).replace(/\s+/g, ' ')

const killed = await rowText('twin')
if (!/killed \(SIGKILL\)/.test(killed)) failures.push(`THE POINT: a killed child does not say so — row reads: ${killed.slice(0, 220)}`)
if (!/nothing here sends that/.test(killed)) failures.push('the row leaves the dashboard itself under suspicion for a signal it never sends')
if (!/its startup output/.test(killed)) failures.push('THE SECOND LIE: a startup-only ring is printed as if it were the last words')
if (/· exited/.test(killed)) failures.push('the old catch-all word is still on a killed row')

const crashed = await rowText('crashed-arm')
if (!/code 1/.test(crashed)) failures.push(`a Python failure hides its exit code: ${crashed.slice(0, 200)}`)
if (/its startup output/.test(crashed)) failures.push('a ring spread over hours was mislabelled as startup output — a wrong claim, worse than none')
if (/SIGKILL/.test(crashed)) failures.push('a crash reads as a kill')

const finished = await rowText('finished-job')
if (!/cleanly/.test(finished)) failures.push(`a job that finished is not said to have finished: ${finished.slice(0, 200)}`)
if (/SIGKILL|code 1/.test(finished)) failures.push('a clean exit borrows another death\'s sentence')

const alive = await rowText('live-arm')
if (/SIGKILL|exited|killed/.test(alive)) failures.push(`a RUNNING child is described as dead: ${alive.slice(0, 200)}`)

await page.screenshot({ path: '/tmp/audit-child-death.png' })
await browser.close()

guard.assertNoEscapes(failures)
if (failures.length) {
  console.log('FAIL  child-death audit')
  for (const f of failures) console.log('  ·', f)
  process.exit(1)
}
console.log('PASS  each death names its own cause, and a startup-only ring says so (4 children compared)')
console.log('      screenshot: /tmp/audit-child-death.png')
