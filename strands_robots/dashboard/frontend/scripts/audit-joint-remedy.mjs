/**
 * Q89 ON THE PAGE: the "your calibration file exists, fix the id" remedy must be READABLE on a card.
 *
 * The remedy this dashboard gives for an `uncalibrated` arm decides whether the operator re-teaches a
 * robot by hand. Q89 made that text correct and, in doing so, made it ~380 characters — and the card
 * renders a joint problem's hint at font-size 10px with no clamp. Correct-but-unreadable is a real
 * failure mode for a sentence whose whole job is to STOP someone doing physical work, so this audit
 * measures the rendered box instead of trusting the CSS.
 *
 * It serves its own /ws/mesh snapshot: one peer, connected, no joints, carrying exactly the
 * annotation the backend now produces. Nothing is spawned and no arm is touched.
 *
 * Run: node scripts/audit-joint-remedy.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const REMEDY = 'Calibration files DO exist on this machine, so this is probably an id/path mismatch rather than '
  + 'an uncalibrated arm: lerobot looks for calibration/robots/<robot_type>/<id>.json, and what exists is '
  + 'robots/so101_follower/follower.json, teleoperators/so101_leader/leader.json. Spawn this arm with the id whose '
  + 'file exists (or copy that file to the path lerobot wants) BEFORE recalibrating - re-teaching an arm that is '
  + 'already calibrated is physical work to fix a filename.'

const peers = {
  'so101-leader': {
    // A REAL silent arm: presence declares the arm family (so expectsJoints says yes) and state is
    // arriving with the joints section omitted — the shape jointAbsence's interesting branch needs.
    peer_id: 'so101-leader', stale: false, role: 'leader', role_basis: 'measured', origin: 'managed',
    presence: { connected: true, robot: 'so101', hw: 'so_follower' },
    state: { t: Math.floor(Date.now() / 1000), joints: {} },
    joint_problem: {
      kind: 'uncalibrated',
      headline: 'this board has no calibration, so its positions cannot be read in degrees',
      remedy: REMEDY,
      detail: "RuntimeError: FeetechMotorsBus(Port '/dev/cu.usbmodem5AB01818061', 6x sts3215) has no calibration registered.",
    },
  },
}

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
await page.routeWebSocket('**/ws/mesh**', ws => {
  ws.send(JSON.stringify({ type: 'snapshot', dashboard_peer_id: 'audit-dash', peers }))
})
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(4000)

const hint = page.locator('.joints.empty .hint').first()
if (await hint.count() === 0) {
  failures.push('the card renders no hint at all for a peer carrying joint_problem — the remedy never reaches the page')
} else {
  const text = (await hint.innerText()).replace(/\s+/g, ' ')
  const box = await hint.boundingBox()
  const size = await hint.evaluate(el => getComputedStyle(el).fontSize)
  // The sentence that stops the physical work must be present, not just the generic tail. This
  // check DOUBLES AS THE FIXTURE-LANDED SENTINEL (Q114): no live peer says "id/path mismatch", so
  // if the ws fixture failed to arrive this audit fails loudly instead of quietly measuring
  // whatever the real fleet happened to be doing.
  if (!/id\/path mismatch/.test(text)) failures.push(`the card's hint omits the mismatch sentence: ${text.slice(0, 200)}`)
  // ...and since Q115 CLAMPS this element to 4 lines in the card, innerText is no longer proof of
  // VISIBILITY: the DOM keeps the whole string while the box shows part of it. The clamp's other
  // half is what makes that safe, so assert it here too — title must carry the entire remedy.
  const title = await hint.getAttribute('title')
  if (!title || !/id\/path mismatch/.test(title)) {
    failures.push('the hint is clamped but its title does not carry the full remedy — the hidden half is unreachable')
  }
  if (/Calibrate this arm/.test(text)) failures.push('the card still tells the operator to recalibrate a calibrated arm')
  // Readability, measured: a hint taller than a third of the card is a wall of text, and 10px is the floor.
  console.log(`hint: ${Math.round(box.width)}x${Math.round(box.height)}px at ${size}, ${text.length} chars`)
  if (parseFloat(size) < 10) failures.push(`hint font-size ${size} is below the 10px readability floor`)
  // 220px was the old ceiling and far too generous: the reason ate 60% of the tile and pushed every
  // sibling down the page while this audit stayed green (Q115). The threshold now comes from the
  // CLAMP'S OWN GEOMETRY rather than taste - 4 lines x 10px x 1.35 = 54px, measured at exactly 54 -
  // with slack for one wrapped line. MEASURED SEPARATION, by un-clamping the live page: 54px
  // clamped vs 122px not. My first attempt put this at 120, which "passed" by 2px; a threshold that
  // close to the failing value is a coincidence, not a test.
  if (box.height > 80) failures.push(`the hint is ${Math.round(box.height)}px tall on a card — the clamp is gone; it needs the drawer, not the card`)
  await page.screenshot({ path: '/tmp/audit-joint-remedy.png' })
}
if (thrown.length) failures.push(`page errors: ${thrown.join(' | ')}`)

await browser.close()
if (failures.length) { console.error('FAIL\n- ' + failures.join('\n- ')); process.exit(1) }
console.log('ok — the Q89 remedy reaches the card, names the mismatch, and fits')
