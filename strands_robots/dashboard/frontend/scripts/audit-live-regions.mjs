/**
 * Q157-Q159: the three announcement regions exist IN THE DOM with the attributes the
 * design decided on — proven in a browser, because a rule that is only unit-tested has
 * never been read by an accessibility tree.
 *
 * The three cases are deliberately DIFFERENT, and this audit's value is refusing to let
 * them drift into one careless copy:
 *   * the agent transcript is a role=log with aria-live OFF (its text arrives one token at
 *     a time, so a live region would stutter the reply for its whole length);
 *   * the activity list is a role=log with aria-live OFF plus a polite atomic sibling that
 *     speaks only the newest line (entries arrive whole, but history loads on open);
 *   * the training screen has a polite atomic region for job transitions.
 * A missing region is a defect; a transcript that has BECOME live is the specific
 * regression this file exists to catch.
 *
 * Read-only: navigates, opens two panels, reads attributes. Clicks nothing that commands a
 * robot. serviceWorkers must be BLOCKED (this dashboard is a PWA — a served-by-worker
 * response is not interceptable and can be stale).
 */
import fs from 'node:fs'
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'

const problems = []
const notes = []

const attrs = (el) => el.evaluate(n => ({
  role: n.getAttribute('role'), live: n.getAttribute('aria-live'),
  label: n.getAttribute('aria-label'), atomic: n.getAttribute('aria-atomic'),
}))

const browser = await chromium.launch()
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(2500)

// --- 1. the agent transcript ---------------------------------------------------------
// The dock starts collapsed; the toggle is labelled by its own state.
const toggle = page.locator('button[aria-label="show the conversation"]')
if (await toggle.count()) await toggle.first().click()
await page.waitForTimeout(600)
const transcript = page.locator('.dock-scroll')
if (!(await transcript.count())) problems.push('agent transcript (.dock-scroll) not found — could not open the dock')
else {
  const a = await attrs(transcript.first())
  if (a.role !== 'log') problems.push(`agent transcript role=${a.role} (want log)`)
  if (a.live !== 'off') problems.push(`REGRESSION: agent transcript aria-live=${a.live} — a per-token stream must not be live`)
  if (!a.label) problems.push('agent transcript has no accessible name')
  else notes.push(`transcript: role=${a.role} live=${a.live} name="${a.label}"`)
  const say = page.locator('.dock-scroll ~ [role="status"], [role="status"][aria-atomic="true"]')
  notes.push(`dock status regions present: ${await say.count()}`)
}

// --- 2. the activity sheet ------------------------------------------------------------
const actBtn = page.locator('button', { hasText: /activity/i })
if (await actBtn.count()) {
  await actBtn.first().click()
  await page.waitForTimeout(900)
  const list = page.locator('ul.activity')
  if (!(await list.count())) notes.push('activity list not rendered (empty feed renders a hint instead) — checking the region only')
  else {
    const a = await attrs(list.first())
    if (a.role !== 'log') problems.push(`activity list role=${a.role} (want log)`)
    if (a.live !== 'off') problems.push(`activity list aria-live=${a.live} — history loads on open, it must not be live`)
    if (!a.label) problems.push('activity list has no accessible name')
    else notes.push(`activity: role=${a.role} live=${a.live} name="${a.label}"`)
  }
  const region = page.locator('.drawer-body [role="status"][aria-live="polite"][aria-atomic="true"]')
  if (!(await region.count())) problems.push('activity sheet has no polite atomic announcement region')
  else notes.push('activity announcement region present')
  await page.keyboard.press('Escape')
  await page.waitForTimeout(400)
} else problems.push('could not find the activity button')

// --- 3. the training screen -----------------------------------------------------------
const tab = page.locator('button', { hasText: /^\s*(🎓\s*)?train/i })
if (await tab.count()) {
  await tab.first().click()
  await page.waitForTimeout(1500)
  const jobs = page.locator('h3', { hasText: /^Jobs$/ })
  if (!(await jobs.count())) notes.push('Jobs heading not found — training screen may not have loaded')
  const region = page.locator('[role="status"][aria-live="polite"][aria-atomic="true"]')
  const n = await region.count()
  if (!n) problems.push('training screen has no polite atomic region for job transitions')
  else notes.push(`training: ${n} polite atomic region(s)`)
} else problems.push('could not find the training tab')

await browser.close()

for (const n of notes) console.log(`  note: ${n}`)
if (problems.length) {
  console.log(`FAIL  ${problems.length} live-region problem(s):`)
  for (const p of problems) console.log(`        ${p}`)
  process.exit(1)
}
console.log('PASS  all three announcement regions present with the intended semantics')
