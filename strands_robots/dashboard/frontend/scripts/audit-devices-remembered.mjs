/**
 * The spawn memory reaches the devices screen — and its button cannot fire on a busy bus (Q41).
 *
 * After a restart `managed` is empty, so two configured arms read as unknown hardware while their
 * whole spawn payload sits in profiles.json. The backend halves are unit-tested; this is the part
 * only a browser can answer, and one assertion here guards a PHYSICAL action: the "spawn again"
 * button must be disabled while something already drives that bus, because two processes on one
 * servo bus is the Q26 collision class with a real arm attached.
 *
 * /api/devices is INJECTED and the spawn route is intercepted (recorded, never forwarded), so this
 * audit cannot start a child process or energise an arm.
 *
 * Run: node scripts/audit-devices-remembered.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const FREE = '/dev/cu.usbmodemFREE1'
const BUSY = '/dev/cu.usbmodemBUSY1'

const doc = {
  serial_ports: [
    { device: FREE, serial_number: 'SERIALFREE', likely_robot: 'so101',
      role: 'follower', role_volts: 12.6, role_source: 'measured',
      remembered: { peer_id: 'so101-arm-1', robot_name: 'so101', mode: 'real',
                    cameras: ['top', 'wrist'], robot_id: 'arm_1', saved_at: 1787115801,
                    // Q43: the wrist index the memory names is blocked by macOS right now. The
                    // operator must read that HERE, not out of a child's log after the arm came up
                    // streaming joints only.
                    camera_health: { ok: false, cameras: [
                      { name: 'top', index: 2, state: 'ready', reason: 'opened just now' },
                      { name: 'wrist', index: 1, state: 'blocked',
                        reason: 'macOS has not granted camera access to this process',
                        remedy: 'start the dashboard from a terminal and allow access' },
                    ], text: 'the saved config names wrist (index 1), which is not available right now: macOS has not granted camera access to this process. Spawning anyway works - the arm drops the camera it cannot open and comes up streaming joints only, which looks healthy and records episodes with no pictures in them' } } },
    // The REAL profile on cagatay's desk: measured 12.6V = follower, saved id says "leader_arm".
    { device: BUSY, serial_number: 'SERIALBUSY', role: 'follower', role_volts: 12.6, role_source: 'measured',
      remembered: { peer_id: 'so101-arm-2', robot_name: 'so101', mode: 'real', cameras: [], robot_id: 'leader_arm' } },
    // A board nobody configured: no `remembered` key at all, and the screen must invent nothing.
    { device: '/dev/cu.usbmodemNEW1', serial_number: 'SERIALNEW' },
  ],
  cameras: [], camera_names: [], camera_problem: null,
  managed: {
    'so101-arm-2': { peer_id: 'so101-arm-2', alive: true, port: BUSY, robot_name: 'so101', pid: 4242 },
  },
}

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
const spawns = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
// ORDER MATTERS, and it cost this script its first run: playwright tries the LAST registered
// matching handler first, so the broad "nothing else may reach the machine" guard goes FIRST and the
// specific routes after it. Registered the other way round, the catch-all swallowed the respawn POST
// and the audit reported "the button did nothing" - blaming the UI for its own plumbing.
await page.route('**/api/devices/**', r => r.fulfill({ status: 200, contentType: 'application/json', body: '{}' }))
await page.route('**/api/devices?**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(doc) }))
await page.route('**/api/devices', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(doc) }))
await page.route('**/api/devices/spawn-remembered', async r => {
  spawns.push(JSON.parse(r.request().postData() ?? '{}'))
  await r.fulfill({ status: 200, contentType: 'application/json',
    body: JSON.stringify({ peer_id: 'so101-arm-1', status: 'running', respawned_from_profile: true }) })
})

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(6000)
await page.locator('button.chip:has-text("devices")').first().click()
await page.waitForTimeout(2500)

// Scoped to the "Servo boards" list: the MANAGED list also names the busy port in its meta line, and
// an unscoped `li` matched that one first - so a passing assertion would have been about the wrong
// element entirely.
const ports = page.locator('section', { has: page.locator('h3:has-text("Servo boards")') }).first()
const rows = (await ports.count()) ? ports.locator('li') : page.locator('li')
const rowFor = (dev) => rows.filter({ hasText: dev }).first()
const freeRow = rowFor(FREE)
const busyRow = rowFor(BUSY)
const newRow = rowFor('/dev/cu.usbmodemNEW1')

// ---- the free board: what it was, in words the operator recognises
if (!(await freeRow.locator('.remembered').count())) {
  failures.push('a board with a saved profile shows no memory of it — the operator is asked to re-type a config that is on disk')
} else {
  const text = await freeRow.locator('.remembered').innerText()
  for (const needle of ['so101-arm-1', 'so101', 'real', 'top + wrist', 'arm_1']) {
    if (!text.includes(needle)) failures.push(`the memory line omits "${needle}"`)
  }
  // Camera INDICES must not appear in the SUMMARY: the saved ones are what macOS renumbers, so
  // printing them as part of "what this board was" is confidently stale. The Q43 notice after the ⚠
  // is allowed to name an index, because that judgment was made against the CURRENT scan — a fresh
  // fact about now, not a remembered one. This distinction is the whole reason the two are separate
  // strings, and the first version of this audit conflated them.
  const summary = text.split('⚠')[0]
  if (/index|\b2\b/.test(summary.replace('arm_1', ''))) failures.push(`the memory summary prints camera indices: ${summary}`)
  const btn = freeRow.locator('.remembered button')
  const label = await btn.innerText()
  if (!label.includes('so101-arm-1')) failures.push(`the button does not name the peer it will start: "${label}"`)
  // A neutral calibration id must not raise a role warning (that warning is asserted on the busy
  // row below); only the camera notice belongs on this row.
  if (/name is what is wrong/.test(text)) failures.push(`a neutral calibration id raised a role warning: ${text.slice(0, 160)}`)
  if (await btn.isDisabled()) failures.push('the free board\'s respawn button is disabled')
  // Q43: the camera trouble is stated where the decision is made, with its consequence and remedy.
  if (!/wrist \(index 1\)/.test(text)) failures.push('the blocked camera is not named on the row')
  if (!/no pictures in them/.test(text)) failures.push('the row states the camera problem without its consequence')
  if (!/start the dashboard from a terminal/.test(text)) failures.push('the remedy never reaches the operator')
  // ...and it must NOT become a gate: dropping a camera is survivable, so the spawn stays offered.
  if (await btn.isDisabled()) failures.push('THE POINT: a camera warning disabled the spawn button — a warning, not a refusal')
}

// ---- THE DANGEROUS ONE: a bus something already drives must not be spawnable again
if (await busyRow.locator('.remembered').count()) {
  const btn = busyRow.locator('.remembered button')
  if (!(await btn.isDisabled())) {
    failures.push('THE DANGEROUS ONE: the respawn button is live on a bus a running child already owns — two processes on one servo bus')
  }
  if (!(await btn.innerText()).includes('already running')) {
    failures.push('the busy row does not say why its button is dead')
  }
} else {
  failures.push('the running board lost its memory line entirely')
}

// ---- a name that contradicts the measurement is called out ON THE ROW, not two clicks away
{
  const text = await busyRow.locator('.remembered').innerText()
  if (!/leader_arm/.test(text)) failures.push('the contradicting calibration id is hidden instead of shown')
  if (!/12\.6V = follower/.test(text)) {
    failures.push(`the row shows a leader-named id under a follower badge with no warning: ${text.replace(/\n+/g, ' | ').slice(0, 200)}`)
  }
  if (!/reuse the memory anyway/.test(text)) failures.push('the warning reads as a refusal instead of a note')
}

// ---- a board nobody configured invents nothing
if (await newRow.locator('.remembered').count()) {
  failures.push('an unconfigured board shows a spawn memory — absence must render as nothing')
}

// ---- the click sends the PORT and nothing else: the payload lives server-side
await freeRow.locator('.remembered button').click()
await page.waitForTimeout(1200)
if (spawns.length !== 1) {
  failures.push(`expected exactly one respawn request, got ${spawns.length}`)
} else {
  const body = spawns[0]
  if (body.port !== FREE) failures.push(`the request carries the wrong port: ${JSON.stringify(body)}`)
  if (Object.keys(body).length !== 1) {
    failures.push(`the client re-sent the payload instead of letting the server hold it: ${JSON.stringify(body)}`)
  }
}
if (await page.locator('.crashcard').count()) failures.push('the devices screen crashed')
if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)

await ctx.close()
await browser.close()

if (failures.length) {
  console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n'))
  process.exit(1)
}
console.log('devices remembered: summary names peer/family/mode/camera NAMES; a blocked saved index is stated with its consequence + remedy and does NOT gate the spawn; a contradicting calibration id is called out; a busy bus refuses; an unconfigured board shows nothing; the click sends only the port')
