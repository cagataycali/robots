/**
 * A session the dashboard died inside REACHES the operator's screen (BUGS.md Q40).
 *
 * The backend half is unit-tested (tests/test_dashboard_record_crash.py), and that is exactly the
 * state Q38 and R5 were in when the fix was correct in the lib and inert in the UI. Only a browser
 * can answer "does the person about to record see that the last take never closed".
 *
 * The session is INJECTED: reproducing it for real means killing the dashboard mid-recording on a
 * rig nobody is standing next to. What is under test is the rendering and, just as much, what the
 * screen does NOT offer — no button one tap from the form may delete an hour of hand-guiding.
 *
 * READ-ONLY: every /api/record call is intercepted, so no session is opened, no dataset is created
 * and no arm is commanded. Safe against the live rig.
 *
 * Run: node scripts/audit-record-interrupted.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import { blockMutations, assertNoEscapes } from './lib/audit-guard.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const NOTICE_TEXT =
  'the dashboard stopped while a recording session was open: “cagatay/so101-pick” was opened about 37 '
  + 'minutes ago, driving so101-arm-1 and so101-arm-2. Episodes already written are on disk; the ones '
  + 'in flight were not flushed, and so101-arm-1 and so101-arm-2 were left despawned - respawn them from devices.'

const idle = (interrupted) => ({
  dataset: null, task: '', leader: null, follower: null, target_episodes: 10,
  phase: 'idle', fps: 30, episodes: [], error: null,
  ...(interrupted ? {
    interrupted: {
      dataset: 'cagatay/so101-pick', task: 'pick up the cube',
      arms: ['so101-arm-1', 'so101-arm-2'], opened_ago: 2220, text: NOTICE_TEXT,
      next: [
        'record into “cagatay/so101-pick” again to continue that dataset (the name is taken, so a new session must use another name)',
        'or delete “cagatay/so101-pick” if the take is worthless',
      ],
    },
  } : {}),
})

const open = async (interrupted) => {
  // serviceWorkers:'block' is REQUIRED: this dashboard is a PWA and a response served by its
  // service worker is not interceptable, so the fixture would never land and the audit would
  // blame the UI for its own plumbing.
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, serviceWorkers: 'block' })
  const page = await ctx.newPage()
  /* The audit hardware guard goes FIRST: playwright matches handlers in REVERSE registration order, so
     every fixture below still wins, and any MUTATING request this audit forgot to intercept is blocked
     and recorded instead of reaching the running dashboard (which spawns processes and commands arms). */
  const guard = await blockMutations(page)
  const thrown = []
  page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
  const body = JSON.stringify(idle(interrupted))
  await page.route('**/api/record/session', r => r.fulfill({ status: 200, contentType: 'application/json', body }))
  await page.route('**/api/record/**', r => r.fulfill({ status: 200, contentType: 'application/json', body }))
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(6000)
  await page.locator('button.chip:has-text("record")').first().click()
  await page.waitForTimeout(2500)
  return { page, ctx, thrown }
}

const browser = await chromium.launch()

// ---- a crash-interrupted session: named, dated, and the arms accounted for
{
  const { page, ctx, thrown } = await open(true)
  const banner = page.locator('.artifact-hold', { hasText: 'recording session was open' }).first()
  if (!(await banner.count())) {
    failures.push('an interrupted session renders nothing - the record screen still shows an empty form over a half-written dataset')
  } else {
    if (!(await banner.isVisible())) failures.push('the interrupted notice is in the DOM but not visible')
    if ((await banner.getAttribute('role')) !== 'status') failures.push('the notice is not role=status')
    const text = await banner.innerText()
    for (const needle of ['cagatay/so101-pick', '37 minutes ago', 'left despawned', 'not flushed']) {
      if (!text.includes(needle)) failures.push(`the notice does not mention "${needle}"`)
    }
    // Both next actions, as WORDS.
    if (!/name is taken/.test(text)) failures.push('the notice does not say the name is taken (Q39 continuity)')
    if (!/if the take is worthless/.test(text)) failures.push('the notice does not offer the delete option in words')
    const buttons = await banner.locator('button').count()
    if (buttons) failures.push(`THE DANGEROUS ONE: the interrupted notice offers ${buttons} button(s) - a tap next to a form must not delete a dataset`)
  }
  // The form is still right there: an interrupted take must not block the next one.
  if (!(await page.locator('.train-form input').first().isVisible())) {
    failures.push('the notice replaced the form instead of sitting above it')
  }
  if (await page.locator('.crashcard').count()) failures.push('the record screen crashed on an interrupted session')
  if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)
  await ctx.close()
}

// ---- an ordinary idle session: silence. A banner that is always on is not a banner.
{
  const { page, ctx, thrown } = await open(false)
  if (await page.locator('.artifact-hold', { hasText: 'recording session was open' }).count()) {
    failures.push('a clean idle session shows the interrupted notice')
  }
  if (thrown.length) failures.push(`page threw (clean): ${thrown.join(' ; ')}`)
  await ctx.close()
}

await browser.close()

assertNoEscapes(failures)
if (failures.length) {
  console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n'))
  process.exit(1)
}
console.log('record interrupted: the notice names the dataset/age/arms, keeps Q39 continuity, offers NO destructive button, leaves the form usable, and stays silent on a clean idle session')
