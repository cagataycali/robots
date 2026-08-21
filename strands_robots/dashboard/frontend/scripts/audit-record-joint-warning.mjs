/**
 * THE RECORD FORM REFUSES AN ARM THAT CANNOT SAY WHERE IT IS — on the page, against the live fleet.
 *
 * 91d1a009 (client) predicts 4839a8b2's 409 (server): a chosen arm whose snapshot is fresh and carries
 * no joint positions cannot be recorded from, because the follower's positions are the dataset's
 * observations and the leader's are its actions. This audit picks the arms out of /api/fleet by whether
 * they ACTUALLY report joints and then drives the real form, so it proves the whole chain — mesh
 * snapshot -> websocket -> Peer.state.joints -> the rule -> the DOM and the disabled button. The unit
 * tests cannot: they hand-build peers, and a snapshot field renamed upstream would leave them green
 * while every arm on the page looked jointless (or none did).
 *
 * The CONTROL matters as much as the warning: an arm that does report joints must clear its slot. A
 * check that only ever fires is indistinguishable from a check that always fires, and this one disables
 * the submit button — a false positive locks the record screen.
 *
 * READ-ONLY: it opens the record screen and changes two <select>s. It never submits the form, so no
 * session is opened, no port claimed, nothing energised — safe with the arms powered.
 *
 * Run: node scripts/audit-record-joint-warning.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const notes = []
const MAX_AGE_S = 30  // lib/recordArms.MAX_AGE_S / record_joints.MAX_AGE_S

const fleet = await fetch(`${BASE}/api/fleet`, { headers: { Authorization: `Bearer ${TOKEN}` } })
  .then(r => r.ok ? r.json() : Promise.reject(new Error(`/api/fleet -> HTTP ${r.status}`)))
const now = Date.now() / 1000
/** The same three questions the rule asks, from the API rather than the browser. */
const classify = (p) => {
  const joints = p?.state?.joints
  const count = p?.state == null ? null
    : joints == null ? 0
    : Array.isArray(joints) ? joints.length
    : typeof joints === 'object' ? Object.keys(joints).length : null
  const age = typeof p?.last_seen === 'number' && p.last_seen > 0 ? Math.max(0, now - p.last_seen) : null
  return { count, age, silent: count === 0 && age !== null && age <= MAX_AGE_S, reads: (count ?? 0) > 0 }
}
const peers = Object.entries(fleet?.peers ?? {}).map(([id, p]) => ({ id, ...classify(p) }))
const silent = peers.find(p => p.silent)
const reading = peers.find(p => p.reads)
notes.push(`note: fleet = ${peers.map(p => `${p.id}(joints=${p.count ?? '?'}, ${p.age === null ? 'undateable' : `${p.age.toFixed(0)}s`})`).join(', ')}`)

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
page.on('pageerror', e => failures.push(`page threw: ${String(e.message).slice(0, 180)}`))
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)
/* TIMING TRAP, and it cost a whole journal entry: the arm <select>s render from peers immediately but
   the MEASURED roles arrive from a separate /api/devices request, so an audit that reads the options a
   couple of seconds after the click sees "role not measured" on an arm that was measured at 12.6V — and
   reports a role bug that does not exist. Wait for the request the labels depend on.
   (Related: do NOT probe an API with page.evaluate(fetch) — the app attaches the bearer token itself, so
   a bare in-page fetch 401s while the app's own request beside it returns 200.) */
const devicesDone = page.waitForResponse(r => r.url().includes('/api/devices') && r.status() === 200, { timeout: 15000 })
await page.locator('button.chip:has-text("record")').first().click()
await devicesDone.catch(() => {})
await page.waitForTimeout(1500)

/* The arm selects are found by their LABEL, not their index: this screen has six selects and the
   policy pickers come first. */
const find = async (re) => {
  const sel = page.locator('select')
  for (let i = 0; i < await sel.count(); i++) {
    const label = await sel.nth(i).evaluate(el => (el.closest('label')?.querySelector('span')?.textContent || '').trim())
    if (re.test(label)) return sel.nth(i)
  }
  return null
}
const leaderSel = await find(/leader/i)
const followerSel = await find(/follower/i)
if (!leaderSel || !followerSel) failures.push('the record form has no leader/follower selects — it did not render')

const warnings = async () => (await page.locator('.train-msg.warn').allInnerTexts()).map(t => t.replace(/\s+/g, ' '))

if (leaderSel && followerSel) {
  if (silent) {
    // Put the silent arm in the FOLLOWER slot: that is the one whose positions become observations.
    await followerSel.selectOption(silent.id)
    await page.waitForTimeout(1200)
    const w = (await warnings()).filter(t => t.includes(silent.id))
    notes.push(`note: "${silent.id}" (${silent.count} joints, ${silent.age.toFixed(0)}s old) -> ${w[0]?.slice(0, 220) ?? '(no warning)'}`)
    if (!w.length) failures.push(`${silent.id} reports no joints in a ${silent.age.toFixed(0)}s-old snapshot and the form says nothing`)
    else {
      if (!/no observations to learn from/.test(w[0])) failures.push('the warning does not say what the dataset would lack')
      if (!/refused/.test(w[0])) failures.push('the warning does not say the recording will be refused, so it reads like a choice')
      if (/\bI (know|understand)\b/i.test(w.join(' '))) failures.push('an acknowledgement is offered for a refusal that has no override flag')
    }
    const btn = page.locator('button.btn.go.wide').first()
    if (!await btn.isDisabled()) failures.push('the submit button is still enabled with a jointless arm selected')
  } else {
    notes.push('note: no arm on this fleet is both fresh and jointless, so the warning could not be exercised')
  }

  if (reading) {
    await followerSel.selectOption(reading.id)
    await page.waitForTimeout(1200)
    const w = (await warnings()).filter(t => t.includes(reading.id))
    notes.push(`note: control "${reading.id}" (${reading.count} joints) -> ${w[0]?.slice(0, 160) ?? 'no warning (correct)'}`)
    // THE CONTROL: a check that always fires is not a check, and this one disables the button.
    if (w.length) failures.push(`${reading.id} reports ${reading.count} joints and is warned about anyway: ${w[0].slice(0, 160)}`)
  } else {
    notes.push('note: no arm on this fleet reports joints, so the control case could not be exercised')
  }
}

await browser.close()
for (const n of notes) console.log(n)
if (failures.length) {
  console.error(`FAIL (${failures.length})`)
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
console.log('PASS  the record form warns about a jointless arm and leaves a reading one alone')
