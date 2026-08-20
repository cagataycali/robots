/**
 * The record screen's empty state names the way out (Q44) — proved in a browser.
 *
 * With zero peers this screen used to say "no arms on the mesh" and stop, while the operator's two
 * boards sat one screen over with a one-click respawn. The verdict logic has unit assertions; what
 * only a browser can answer is whether it is WIRED: whether the mesh really reports zero peers to
 * this component, whether the sentence reaches the DOM, and whether the offered button actually
 * opens the devices screen instead of being decoration.
 *
 * The /ws/mesh socket is served BY THIS SCRIPT with an empty snapshot, and /api/devices is injected —
 * so this audit reads nothing from the live fleet and can neither spawn nor move anything.
 *
 * Run: node scripts/audit-record-no-arms.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const devicesDoc = {
  serial_ports: [
    { device: '/dev/cu.usbmodemAAA1', serial_number: 'AAA', remembered: { peer_id: 'so101-arm-1', cameras: [] } },
    { device: '/dev/cu.usbmodemBBB1', serial_number: 'BBB', remembered: { peer_id: 'so101-arm-2', cameras: [] } },
  ],
  cameras: [], camera_names: [], camera_problem: null, managed: {},
}

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))

// An EMPTY fleet, served locally: the live mesh has two arms on it, and this screen's whole point is
// what it says when it has nothing.
await page.routeWebSocket('**/ws/mesh**', ws => {
  ws.send(JSON.stringify({ type: 'snapshot', dashboard_peer_id: 'audit-dash', peers: {} }))
})
await page.route('**/api/devices**', r => r.fulfill({
  status: 200, contentType: 'application/json', body: JSON.stringify(devicesDoc) }))
// The record screen also asks for known datasets; keep that off the real disk.
await page.route('**/api/training/datasets**', r => r.fulfill({
  status: 200, contentType: 'application/json', body: JSON.stringify({ datasets: [] }) }))

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)

// ---- Q45: the HOME screen, before any panel is opened, must already name the route. This is the
// first thing an operator sees after a restart, and it used to offer only a python snippet.
{
  const home = await page.locator('body').innerText()
  if (!/devices screen remembers so101-arm-1 and so101-arm-2/.test(home)) {
    failures.push(`the home empty state does not name the remembered boards: ${home.slice(0, 400).replace(/\n+/g, ' | ')}`)
  }
  // The heading already says the mesh is empty - saying it twice reads like a stutter.
  if ((home.match(/no arms are on the mesh/gi) ?? []).length) {
    failures.push('the home screen repeats the absence its own heading states')
  }
  // The snippet stays: someone with no board at all still needs it. It just is not the only answer.
  if (!/from strands_robots import Robot/.test(home)) failures.push('the home screen lost the start snippet')
}
await page.locator('button.chip:has-text("record")').first().click()
await page.waitForTimeout(2500)

const sheet = page.locator('.sheet, .rec, .overlay').filter({ hasText: 'follower' }).first()
const body = (await page.locator('body').innerText())

// ---- the sentence, with the boards NAMED
if (!/no arms are on the mesh/.test(body)) {
  failures.push(`the empty record screen does not say the fleet is empty: ${body.slice(0, 300).replace(/\n+/g, ' | ')}`)
}
for (const name of ['so101-arm-1', 'so101-arm-2']) {
  if (!body.includes(name)) failures.push(`the remembered board ${name} is not named on the record screen`)
}
if (!/one click there/.test(body)) failures.push('the route to bringing a board back is not stated')
// The OLD dead-end sentence must be gone, not merely supplemented.
if (/no arms on the mesh(?!\s*,)/.test(body) && !/devices screen remembers/.test(body)) {
  failures.push('the dead-end wording is still what the operator reads')
}

// ---- the offer is real: it must open the devices screen
const btn = page.locator('button:has-text("open the devices screen")').first()
if (!(await btn.count())) {
  failures.push('no button offers the devices screen')
} else {
  await btn.click()
  await page.waitForTimeout(1500)
  const after = await page.locator('body').innerText()
  // Case-INSENSITIVE on purpose: these headings are uppercased by CSS text-transform, and innerText
  // returns the transformed text — a case-sensitive match here failed on a screen that had opened
  // perfectly, i.e. the audit's first red was the audit's own bug.
  if (!/servo boards|detected hardware|managed robots/i.test(after)) {
    failures.push('the offered button does not actually open the devices screen — it is decoration')
  }
  // And the boards it promised are there, so the sentence was not a lie.
  if (!/so101-arm-1/.test(after)) failures.push('the devices screen it opens does not show the board it named')
}

if (await page.locator('.crashcard').count()) failures.push('a screen crashed')
if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)

await ctx.close()
await browser.close()

if (failures.length) {
  console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n'))
  process.exit(1)
}
console.log('no-arms route: the HOME screen names the remembered boards without repeating its own heading, and the record screen reads "no arms are on the mesh, but the devices screen remembers so101-arm-1 and so101-arm-2 — one click there…", and the offered button really opens that screen')
