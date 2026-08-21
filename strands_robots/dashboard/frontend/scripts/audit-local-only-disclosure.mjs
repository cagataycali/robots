/**
 * The run form TELLS the operator which knobs the mesh cannot carry.
 *
 * `registry/policies.json` marks some provider kwargs `unsettable_over_mesh` — lerobot_local marks
 * seven, including `norm_tag`. The wire schema drops them, so a value typed for one of those would be
 * collected and silently discarded, and a precheck built on one can never fire from this screen
 * (4b76b962 removed exactly such a branch). RunForm handles this honestly: it renders a disclosure
 * naming the dropped keys. But only the pure summary function is tested — delete the JSX block and
 * every test still passes while the operator loses the disclosure entirely.
 *
 * So this audit checks the rendered DOM: the sentence exists, and it NAMES the keys rather than
 * gesturing at "some options". It also asserts the inverse — a provider with nothing unsettable must
 * render no such line, because a disclosure that appears always is noise the operator learns to skip.
 *
 * Fully stubbed (ws snapshot + /api/policies + /api/devices), so it never touches the live fleet:
 * nothing is spawned, recorded or moved, and it is safe with the arms powered.
 *
 * Run: node scripts/audit-local-only-disclosure.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const DROPPED = ['device', 'actions_per_step', 'norm_tag', 'image_keys', 'inference_action_mode', 'embodiment', 'strict_keys']
const provider = (name, unsettable) => ({
  name, description: `${name} (audit stub)`, requires: ['pretrained_name_or_path'],
  config_keys: ['pretrained_name_or_path', ...unsettable], defaults: {}, shorthands: [], url_patterns: [],
  extra: null, trainable: true, wire_safe: true, unsettable_over_mesh: unsettable,
  wire_fields: [{ key: 'pretrained_name_or_path', wire_key: 'pretrained_name_or_path', type: 'string', required: true }],
})
const catalog = [provider('lerobot_local', DROPPED), provider('audit_all_wire', [])]

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1100 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))

await page.routeWebSocket('**/ws/mesh**', ws => {
  ws.send(JSON.stringify({
    type: 'snapshot', dashboard_peer_id: 'audit-dash',
    peers: { 'audit-arm': { peer_id: 'audit-arm', last_seen: Date.now() / 1000, stale: false, origin: 'external',
      state: { peer_id: 'audit-arm', t: Date.now() / 1000, joints: { shoulder_pan: 0.1, shoulder_lift: -0.2 } } } },
  }))
})
/* The run form's schema arrives inside the app-wide /api/config document, so the stub REPLACES only
   its `policies` array and leaves every other field genuine - a hand-built config doc would test a
   shape this server may not even serve. `mock` keeps its real place as the default selection. */
const realConfig = await (await fetch(`${BASE}/api/config`, { headers: { Authorization: `Bearer ${TOKEN}` } })).json()
const stubbed = { ...realConfig, policies: [provider('mock', []), ...catalog] }
await page.route('**/api/config**', r => r.fulfill({
  status: 200, contentType: 'application/json', body: JSON.stringify(stubbed) }))
await page.route('**/api/devices**', r => r.fulfill({
  status: 200, contentType: 'application/json',
  body: JSON.stringify({ serial_ports: [], cameras: [], camera_names: [], camera_problem: null, managed: {} }) }))

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(4000)

/** Reveal the run form: it lives on the robot card / detail behind whatever affordance the UI uses. */
/** An operator picks the provider; the disclosure must follow that choice, not the page load. */
/* Addressed by the aria-label the form declares, not by option text: the options render as operator
   prose ("LeRobot - local checkpoint (runs here)"), so matching the provider's raw name finds nothing
   even when the catalog arrived - the first version of this audit failed exactly there. */
const select = page.getByLabel(/Policy/i).first()
if (!(await select.count())) {
  failures.push('no provider selector on the run form, so the stubbed catalog never reached the screen')
}
if (await select.count()) {
  await select.selectOption({ value: 'lerobot_local' }).catch(e => failures.push(`could not select lerobot_local: ${e.message.slice(0, 80)}`))
  await page.waitForTimeout(1200)
}
/* The disclosure lives inside the provider-options panel (the ⚙ toggle), which is the right place:
   the dropped knobs are themselves in that panel, so an operator who never opens it never types one.
   The audit therefore asserts the contract as it really is - absent until the panel is open, present
   and specific once it is - rather than demanding a line the design deliberately keeps contextual. */
const closedPanel = await page.locator('.local-only').count()
if (closedPanel) {
  failures.push('the wire-drop disclosure renders with the options panel CLOSED, ahead of the fields it '
    + 'describes - if that is now intended, this audit is the thing to update')
}
await page.locator('button[title="Provider options"]').first().click({ timeout: 3000 }).catch(
  e => failures.push(`could not open the provider options panel: ${e.message.slice(0, 80)}`))
await page.waitForTimeout(900)
const shown = (await page.locator('.local-only').count()) > 0
if (process.env.AUDIT_DEBUG) {
  console.error('DEBUG selects:', await page.locator('select').count(),
    '| options:', (await page.locator('option').allInnerTexts()).slice(0, 12).join(','),
    '| local-only:', await page.locator('.local-only').count())
}
if (!shown) {
  failures.push('no local-only disclosure rendered anywhere for a provider with 7 unsettable keys '
    + `(page text: ${(await page.locator('body').innerText()).slice(0, 260).replace(/\n+/g, ' | ')})`)
} else {
  const text = await page.locator('.local-only').first().innerText()
  if (!/built on the robot itself|cannot carry/i.test(text)) {
    failures.push(`the disclosure does not say WHY those options are unavailable: ${text.slice(0, 180)}`)
  }
  for (const key of ['norm_tag', 'image_keys', 'strict_keys']) {
    if (!text.includes(key)) failures.push(`the disclosure never names ${key}, so the operator cannot tell which knob is dropped: ${text.slice(0, 200)}`)
  }
}
/* The inverse, and the reason this cannot just check "some text exists": a provider with nothing
   unsettable must render NO such line, or a disclosure that always appears becomes noise. */
await page.locator('select').first().selectOption({ value: 'audit_all_wire' }).catch(() => {})
await page.waitForTimeout(900)
if (await page.locator('.local-only').count()) {
  failures.push('the disclosure shows for a provider whose every field rides the wire - a line that '
    + 'appears regardless of the provider tells the operator nothing')
}
if (thrown.length) failures.push(`page errors: ${thrown.join(' | ')}`)

await browser.close()
if (failures.length) { console.error('FAIL\n - ' + failures.join('\n - ')); process.exit(1) }
console.log('PASS the run form names the knobs the mesh cannot carry')
