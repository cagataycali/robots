/**
 * A refused recording is ANSWERABLE on the page — proved in a browser (877e08cb).
 *
 * The three camera gates at /api/record/open each refuse with 409 + the name of the flag that
 * proceeds anyway. lib/recordRefusal has unit assertions for which flag a sentence offers; what
 * only a browser can answer is whether the loop CLOSES: does the refusal reach the DOM, does a
 * tick appear under it, and does pressing start again actually put that flag in the request?
 * A "continuable" refusal whose continuation never reaches the wire is the exact bug this arc
 * was about — a helpful message with no door.
 *
 * Everything is injected: the mesh snapshot is served by this script, /api/record/open is
 * intercepted and NEVER reaches the recorder. So this audit cannot open a session, cannot park
 * an arm, and cannot move anything — it only reads what the page sends.
 *
 * Run: node scripts/audit-record-refusal.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
// Q130: the same audit at phone size. VIEWPORT=390x844 checks that a refusal an operator must
// ANSWER is reachable on the screen they actually hold — the record form is the densest one in the
// dashboard, and a tick that needs a horizontal scroll to find is a tick nobody ticks.
const [VW, VH] = (process.env.VIEWPORT ?? '1280x1100').split('x').map(Number)

// The server's real sentence, copied from camera_liveness.drift_refusal. Deliberately not
// paraphrased: the page's tick is derived from these words, so a reworded fixture would prove
// nothing about the shipped copy.
const DRIFT =
  'so101-arm-2: 1 configured camera changed hands - wrist index 0 was USB2.0_CAM1, now Logi 4K Pro '
  + '(USB2.0_CAM1 is at index 1 now). Point the camera at its new index, or pass '
  + 'ignore_camera_identity to record with the index as it stands.'
const TWO_FAULTS = DRIFT + ' Also: pass ignore_dead_cameras to record anyway.'

// The peer shape the dashboard actually renders from (/ws/mesh snapshot): presence carries the
// connection and robot type, and the joints live under `state`. Copied from audit-camera-dropped —
// a fixture in the wrong shape makes this screen say "no arms are on the mesh", which reads like a
// UI bug and is really an audit bug.
const now = Date.now() / 1000
const arm = (id, extra) => ({
  peer_id: id, last_seen: now, stale: false, cameras: {}, origin: 'managed',
  presence: { connected: true, hostname: 'Mac', robot_type: 'so101', timestamp: now },
  // The joints must live UNDER state.joints (lib/recordArms.jointCount reads peer.state.joints) and be
  // datable from last_seen: flat 'shoulder_pan.pos' keys count as ZERO joints, which is a legitimate,
  // unackable refusal ("the episodes would carry no actions to learn from") added after this audit was
  // written. The form then never enables and the failure reads like a UI bug — audit-record-joint-warning
  // has the shape right, and this one had drifted from it.
  state: {
    joints: {
      shoulder_pan: 1.5, shoulder_lift: 0.2, elbow_flex: 0.1,
      wrist_flex: 0.0, wrist_roll: 0.0, gripper: 12,
    },
  },
  ...extra,
})

const peers = {
  'so101-arm-1': arm('so101-arm-1', { role: 'leader', role_basis: 'measured', voltage: 7.6 }),
  'so101-arm-2': arm('so101-arm-2', {
    role: 'follower', role_basis: 'measured', voltage: 12.6,
    cameras: { wrist: { t: now, index_or_path: 0 } },
  }),
}

const IDLE = {
  dataset: null, task: null, leader: null, follower: null, target_episodes: 0,
  phase: 'idle', episodes: [], error: null, fps: 30,
}

const browser = await chromium.launch()

/** Open the record screen with /api/record/open answering `detail` as a 409, and return the page. */
async function recordScreen(detail, vp = { width: VW, height: VH }) {
  const ctx = await browser.newContext({ viewport: vp, serviceWorkers: 'block' })
  const page = await ctx.newPage()
  const thrown = []
  const sent = []
  page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
  await page.routeWebSocket('**/ws/mesh**', ws => {
    ws.send(JSON.stringify({
      type: 'snapshot', dashboard_peer_id: 'gateway-audit', mesh: { connected: true }, t: now, peers,
    }))
  })
  // ORDER MATTERS: playwright matches routes in REVERSE registration order, so the catch-all goes
  // FIRST and the specific ones after it. Registered the other way round (the obvious way), the
  // catch-all swallowed /api/record/open, the 409 never happened, and the audit reported "the
  // refusal never reaches the screen" — blaming the UI for a fixture that outranked itself.
  // The catch-all also guarantees this audit cannot reach a real record verb even if a future
  // panel POSTs on mount.
  await page.route('**/api/record/**', r => r.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify(IDLE) }))
  await page.route('**/api/record/session**', r => r.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify(IDLE) }))
  await page.route('**/api/record/open**', async r => {
    sent.push(JSON.parse(r.request().postData() ?? '{}'))
    await r.fulfill({ status: 409, contentType: 'application/json', body: JSON.stringify({ detail }) })
  })
  // The measured roles come from /api/devices, not the mesh — without this the pickers stay
  // unopinionated ("role not measured") and the start button is correctly disabled, which the
  // first run of this script misread as a UI defect.
  await page.route('**/api/devices**', r => r.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({
      serial_ports: [
        { device: '/dev/cu.usbmodemAAA1', serial_number: 'AAA', remembered: { peer_id: 'so101-arm-1', cameras: [] } },
        { device: '/dev/cu.usbmodemBBB1', serial_number: 'BBB', remembered: { peer_id: 'so101-arm-2', cameras: [] } },
      ],
      cameras: [], camera_names: [], camera_problem: null,
      managed: {
        'so101-arm-1': { peer_id: 'so101-arm-1', alive: true, port: '/dev/cu.usbmodemAAA1', role: 'leader', role_volts: 7.6 },
        'so101-arm-2': { peer_id: 'so101-arm-2', alive: true, port: '/dev/cu.usbmodemBBB1', role: 'follower', role_volts: 12.6 },
      },
    }) }))
  await page.route('**/api/training/datasets**', r => r.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ datasets: [] }) }))
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(4000)
  await page.locator('button.chip:has-text("record")').first().click()
  await page.waitForTimeout(2000)
  return { ctx, page, thrown, sent }
}

/** Fill the form the way an operator does, then press start. */
async function attemptRecording(page) {
  await page.locator('.field input').first().fill('audit/refusal-probe')
  const inputs = page.locator('.field input, .field textarea')
  await inputs.nth(1).fill('pick up the cube')
  // fps is empty by default and an empty rate is a problem the form refuses on (Q54) — a dataset
  // stamped with a rate nobody chose is the bug that field exists to prevent.
  await inputs.nth(2).fill('30')
  // The pickers prefill from the measured roles, but selecting them explicitly keeps this audit
  // about the refusal rather than about pairArms (which has its own tests).
  const picks = page.locator('.field select')
  // The options arrive with the mesh snapshot, so wait for one rather than selecting into an empty
  // picker (which silently leaves the slot blank and disables the button for the wrong reason).
  await page.waitForFunction(
    () => [...document.querySelectorAll('.field select option')].some(o => o.value === 'so101-arm-2'),
    null, { timeout: 8000 }).catch(() => {})
  await picks.nth(0).selectOption('so101-arm-1').catch(() => {})
  await picks.nth(1).selectOption('so101-arm-2').catch(() => {})
  await page.waitForTimeout(300)
  // SCOPED to the form on purpose: `button.go` alone also matches the agent chat's ▶ Run button,
  // which is disabled with an empty prompt — the first version of this audit spent its clicks on it.
  const start = page.locator('.train-form button.go').first()
  // WAIT for the button rather than reading it once: every field change re-renders this form, and
  // the first version of this audit tested `isDisabled()` in the same tick as the last fill and
  // reported a disabled button that was already enabled by the time it printed the evidence.
  try {
    await start.waitFor({ state: 'visible', timeout: 5000 })
    await page.waitForFunction(
      () => !document.querySelector('.train-form button.go')?.disabled, null, { timeout: 5000 })
  } catch {
    const why = JSON.stringify(await page.locator('.train-form').evaluate(f => ({
      inputs: [...f.querySelectorAll('input')].map(i => [i.placeholder, i.value]),
      selects: [...f.querySelectorAll('select')].map(x => x.value),
      acks: [...f.querySelectorAll('.ackrow')].map(a => [a.innerText.slice(0, 60), a.querySelector('input')?.checked]),
      // A disabled primary action is only debuggable if the page SAYS why. Capture what it says:
      // the button's own hint, plus every warning/problem row in the form. If these come back empty
      // the finding is not "the audit is stale", it is "the button refuses in silence".
      hint: f.querySelector('.rec-hint')?.innerText?.slice(0, 200) ?? null,
      warnings: [...f.querySelectorAll('.train-msg, .warn, .problem')].map(x => x.innerText.slice(0, 120)).filter(Boolean),
    })))
    failures.push(`the start button stays disabled with a filled form: ${why}`)
    return false
  }
  await start.click()
  await page.waitForTimeout(1200)
  return true
}

// ---- 1. the refusal reaches the page, and brings its answer with it
{
  const { ctx, page, thrown, sent } = await recordScreen(DRIFT)
  if (await attemptRecording(page)) {
    if (sent.length !== 1) failures.push(`start sent ${sent.length} open requests, expected 1`)
    // The FIRST attempt must never carry an override: consent cannot precede the question.
    if (sent[0] && 'ignore_camera_identity' in sent[0]) {
      failures.push('the first attempt already carried the override — a default consent')
    }
    const body = await page.locator('body').innerText()
    if (!/changed hands/.test(body)) {
      failures.push(`the 409 never reaches the screen: ${body.slice(-400).replace(/\n+/g, ' | ')}`)
    }
    // The server's own explanation, not a rewrite: the operator needs the index that moved.
    if (!/USB2\.0_CAM1 is at index 1 now/.test(body)) {
      failures.push('the refusal on screen drops the server sentence that says where the camera went')
    }
    const tick = page.locator('.ackrow input[type=checkbox]').last()
    const ackText = await page.locator('.ackrow').last().innerText().catch(() => '')
    if (!(await tick.count())) {
      failures.push('a refusal naming ignore_camera_identity offers NO tick — continuable only by curl')
    } else {
      // The tick must state the claim in the first person AND the cost of being wrong: this
      // override permits something that looks perfectly healthy.
      if (!/really are at these indices/.test(ackText)) {
        failures.push(`the tick does not state the operator's claim: ${ackText.replace(/\n+/g, ' | ')}`)
      }
      if (!/WRONG view/.test(ackText)) {
        failures.push(`the tick hides the cost of being wrong: ${ackText.replace(/\n+/g, ' | ')}`)
      }
      // ---- 2. and ticking it puts the flag ON THE WIRE
      await tick.check()
      await page.locator('.train-form button.go').first().click()
      await page.waitForTimeout(1200)
      if (sent.length !== 2) {
        failures.push(`the retry did not reach /api/record/open (${sent.length} requests total)`)
      } else if (sent[1].ignore_camera_identity !== true) {
        failures.push(`the retry omits the flag the refusal named: ${JSON.stringify(sent[1])}`)
      }
      // It must send the ONE flag it was offered, not a blanket bypass.
      if (sent[1] && ('ignore_dead_cameras' in sent[1] || 'ignore_missing_cameras' in sent[1])) {
        failures.push(`the retry sent overrides nobody offered: ${JSON.stringify(sent[1])}`)
      }
    }
  }
  if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)
  if (await page.locator('.crashcard').count()) failures.push('a screen crashed')
  await ctx.close()
}

// ---- 3. two faults refused at once offer NO tick: one box would collect consent for the
// fault the operator did not read.
{
  const { ctx, page, thrown } = await recordScreen(TWO_FAULTS)
  if (await attemptRecording(page)) {
    const body = await page.locator('body').innerText()
    if (!/changed hands/.test(body)) failures.push('the two-fault refusal never reached the screen')
    if (await page.locator('.ackrow input[type=checkbox]').count()) {
      failures.push('a refusal naming TWO overrides still offers a single tick')
    }
  }
  if (thrown.length) failures.push(`page threw (two-fault): ${thrown.join(' ; ')}`)
  await ctx.close()
}

// ---- 4. Q130: on a phone, is the tick FINDABLE? Playwright scrolls a target into view before it
// clicks, so section 1 passing at 390x844 proves the tick EXISTS on a phone, not that a human can
// see it. The mobile failure to fear is horizontal overflow: a tick parked past the right edge is
// reached by a scroll gesture nobody thinks to make on a form that already scrolls vertically.
// Unconditional: run-audits.mjs passes no env, so a phone check gated behind VIEWPORT= would never
// run in the sweep that matters. It opens its own 390x844 context instead.
const PHONE = { width: 390, height: 844 }
{
  const { ctx, page } = await recordScreen(DRIFT, PHONE)
  if (await attemptRecording(page)) {
    const over = await page.evaluate(() => {
      const d = document.documentElement
      return { scroll: d.scrollWidth, client: d.clientWidth }
    })
    // 1px of slack: sub-pixel layout rounding is not a defect.
    if (over.scroll > over.client + 1) {
      failures.push(`the record screen scrolls SIDEWAYS at ${PHONE.width}px (${over.scroll} > ${over.client}) — `
        + 'a refusal answered by a tick off the right edge is a refusal nobody answers')
    }
    const box = await page.locator('.ackrow').first().boundingBox()
    if (!box) failures.push('the consent row has no box at phone width')
    else if (box.x < 0 || box.x + box.width > PHONE.width + 1) {
      failures.push(`the consent row sits outside the ${PHONE.width}px viewport (x=${Math.round(box.x)}, `
        + `w=${Math.round(box.width)})`)
    }
    // The tick must still SAY what it admits at this width: a CLIPPED sentence is consent collected
    // for words the operator could not read. Checked as clipping, not as a keyword — the first
    // version of this looked for "identity"/"changed hands" in a label that reads "my cameras really
    // are at these indices — record with them as they stand", i.e. it invented a defect out of its
    // own wrong expectation. Wrapping is fine; overflow being hidden is not.
    const clip = await page.locator('.ackrow').first().evaluate(el => ({
      text: el.innerText,
      clippedX: el.scrollWidth > el.clientWidth + 1,
      clippedY: el.scrollHeight > el.clientHeight + 1,
      overflow: getComputedStyle(el).overflow,
    }))
    if (clip.clippedX || clip.clippedY) {
      failures.push(`the consent row's words are CLIPPED at ${PHONE.width}px (overflow: ${clip.overflow}) — `
        + 'consent collected for a sentence the operator cannot finish reading')
    }
    if (!/cameras/i.test(clip.text)) {
      failures.push(`the consent row lost its subject at phone width: ${JSON.stringify(clip.text.slice(0, 90))}`)
    }
  }
  await ctx.close()
}

await browser.close()

if (failures.length) {
  console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n'))
  process.exit(1)
}
console.log('record refusal: a 409 naming ignore_camera_identity reaches the DOM with the server\'s own words, '
  + 'is answerable at 390x844 without a sideways scroll or a clipped sentence, '
  + 'grows one tick that states the claim AND the cost, and the retry carries exactly that flag — '
  + 'while a refusal naming two overrides grows none')
