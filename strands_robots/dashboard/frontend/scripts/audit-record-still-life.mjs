/**
 * The still-life warning (BUGS.md Q35) actually REACHES the operator's screen.
 *
 * The judgment is unit-tested twice over (record_motion.py, and the worker's session payload
 * in tests/test_dashboard_record_still_life.py), and that is precisely the state R5 was in
 * when its fix was correct in the lib and inert in the UI for a commit. Nothing in this repo
 * renders a component in a test, so only a browser can answer "does the person holding the
 * leader arm see it".
 *
 * The session is INJECTED, not produced by hardware: freezing a real arm mid-episode means
 * cutting a 12V supply on a rig nobody is standing next to, and the point under test here is
 * the rendering, not the physics. Both directions are checked - the notice present, and a
 * healthy session showing NOTHING (a warning that is always on is not a warning).
 *
 * READ-ONLY: every /api/record call is intercepted, so no session is opened, no dataset is
 * created and no arm is commanded. Safe against the live rig.
 *
 * Run: node scripts/audit-record-still-life.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const STILL_MESSAGE =
  'the follower has not moved for 12s - 354 frames of one unchanging pose (largest joint travel '
  + '0.10 deg). If that is deliberate, ignore this. If it is not: a Feetech bus still REPORTS '
  + 'positions from the USB logic rail when the 12V supply is off, so a tripped supply looks '
  + 'exactly like this - valid numbers, full frame rate, an arm that never moves. Check the '
  + "follower's power before collecting more, and redo this episode."

const session = (motion) => ({
  dataset: 'cagatay/so101-pick', task: 'pick up the cube',
  leader: 'so101-arm-1', follower: 'so101-arm-2', target_episodes: 3,
  phase: 'recording', fps: 30, fps_achieved: 29.8, fps_notice: null, camera_notice: null,
  episodes: [{ index: 0, frames: 354, duration_s: 11.8, thumbnails: {}, status: 'recording' }],
  error: null,
  motion_notice: motion
    ? { still: true, seconds: 11.8, samples: 40, max_travel_deg: 0.104,
        quietest_joint: 'wrist_roll.pos', message: STILL_MESSAGE }
    : null,
})

const open = async (motion) => {
  // serviceWorkers:'block' is REQUIRED, not hygiene: this dashboard is a PWA, and a response
  // served by its service worker is not interceptable by page.route. Without it the injected
  // session never lands and the audit reports "no banner" - i.e. it would blame the UI for a
  // fixture that never arrived. (Cost me the first run of this script.)
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, serviceWorkers: 'block' })
  const page = await ctx.newPage()
  const thrown = []
  page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
  await page.route('**/api/record/session', r => r.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify(session(motion)),
  }))
  // Nothing else on /api/record may be reached: this audit must not be able to open a real
  // session even if a future panel decides to POST on mount.
  await page.route('**/api/record/**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(session(motion)) }))
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(6000)
  await page.locator('button.chip:has-text("record")').first().click()
  await page.waitForTimeout(3000)
  return { page, ctx, thrown }
}

const browser = await chromium.launch()

// ---- a frozen follower: the operator must be told, and the COUNTERS must carry the doubt
{
  const { page, ctx, thrown } = await open(true)
  const banner = page.locator('.rec-motion-notice').first()
  if (!(await banner.count())) {
    failures.push('a still-life session renders no warning at all')
  } else {
    if (!(await banner.isVisible())) failures.push('the still-life warning is in the DOM but not visible')
    if ((await banner.getAttribute('role')) !== 'alert') {
      failures.push('the warning is not role=alert - it changes what the operator does next')
    }
    const text = await banner.innerText()
    for (const needle of ['12V', 'logic rail', 'redo', 'deliberate']) {
      if (!text.includes(needle)) failures.push(`the warning does not mention "${needle}"`)
    }
  }
  // The rate pair is where the lie lives: a frozen arm produces a PERFECT rate, so the
  // numbers themselves have to carry the doubt, not only a box above them.
  const counter = await page.locator('.rec-counter').first().innerText().catch(() => '')
  if (!counter.includes('not moving')) {
    failures.push(`the fps pair does not say "not moving": ${counter.replace(/\n+/g, ' | ').slice(0, 120)}`)
  }
  if (await page.locator('.crashcard').count()) failures.push('the record screen crashed with a still-life session')
  if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)
  await ctx.close()
}

// ---- a healthy session: silence. A warning that is always on is not a warning, and
// motion_notice absent means "nothing to say OR not enough evidence yet" - never reassurance.
{
  const { page, ctx, thrown } = await open(false)
  if (await page.locator('.rec-motion-notice').count()) {
    failures.push('a healthy session shows the still-life warning')
  }
  const counter = await page.locator('.rec-counter').first().innerText().catch(() => '')
  if (counter.includes('not moving')) failures.push('a healthy session says "not moving" on the fps pair')
  if (counter.includes('moving')) failures.push('a healthy session claims the arm IS moving - absence of evidence is not evidence')
  if (thrown.length) failures.push(`page threw (healthy): ${thrown.join(' ; ')}`)
  await ctx.close()
}

await browser.close()
if (failures.length) { console.log('FAILURES:'); for (const f of failures) console.log(`  ✗ ${f}`); process.exit(1) }
console.log('still-life warning: shown with its cause when the arm is frozen, silent when it is not')
