/**
 * A TELEOPERATOR CALIBRATION NEVER GETS A GREEN TICK ON THE PAGE.
 *
 * The defect this proves gone (dde8199b): typing `leader` into the spawn form's calibration id with
 * so101 selected rendered "✓ matches leader (so101_leader, 6 motors)". Green, confident, and about the
 * exact id measured live producing an arm with presence connected:true and ZERO joints — lerobot loads
 * robots/<type>/<id>.json, `leader` exists only as teleoperators/so101_leader/leader.json, and it
 * refused with "has no calibration registered" while the reason sat in a child log.
 *
 * Why an audit and not just the unit tests: the verdict is computed from `/api/calibration`'s parsed
 * listing, so the SIDE has to survive server text -> parseCalibrationList -> entries -> verdict. The
 * unit tests hand-build entries with `deviceType: 'teleoperators'`; only the page can show that the
 * real listing on this machine still carries the field at all (a parse that dropped it would make
 * every id look robot-side and restore the green tick with all tests green).
 *
 * READ-ONLY: it switches the spawn form to `real`, types into a text field, then puts the form back.
 * It never clicks spawn, never touches a port, never moves anything — safe with the arms powered.
 *
 * Run: node scripts/audit-calibration-side.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const notes = []

/* Which ids exist on which side, from the API the page itself reads. The audit needs a REAL
   teleoperator-only id to be meaningful: on a machine that has none, it says so and passes rather
   than inventing one, because a fixture would only test the unit tests again. */
const listing = await fetch(`${BASE}/api/calibration`, { headers: { Authorization: `Bearer ${TOKEN}` } })
  .then(r => r.ok ? r.json() : Promise.reject(new Error(`/api/calibration -> HTTP ${r.status}`)))
const text = String(listing?.text ?? '')
const sides = new Map()  // id -> Set(deviceType)
{
  let type = ''
  for (const line of text.split('\n')) {
    const t = /^#+\s*(robots|teleoperators)\b/i.exec(line.replace(/[*`]/g, '').trim())
    if (t) { type = t[1].toLowerCase(); continue }
    const id = /^\s*[-*]\s+`?([A-Za-z0-9_.-]+)`?/.exec(line)
    if (id && type) sides.set(id[1], (sides.get(id[1]) ?? new Set()).add(type))
  }
}
const teleopOnly = [...sides].find(([, s]) => s.has('teleoperators') && !s.has('robots'))?.[0] ?? null
const robotSide = [...sides].find(([, s]) => s.has('robots'))?.[0] ?? null
notes.push(`note: teleoperator-only id on this machine: ${teleopOnly ?? 'none'}; robot-side id: ${robotSide ?? 'none'}`)

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
page.on('pageerror', e => failures.push(`page threw: ${String(e.message).slice(0, 180)}`))
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)
await page.locator('button.chip:has-text("devices")').first().click()
await page.waitForTimeout(3500)

/* The mode control is found by its OPTIONS, not its position: this drawer has seven selects and the
   spawn form is not the first of them. */
let mode = null
const selects = page.locator('select')
for (let i = 0; i < await selects.count(); i++) {
  const vals = await selects.nth(i).locator('option').evaluateAll(o => o.map(x => x.value))
  if (vals.includes('real') && vals.includes('sim')) { mode = selects.nth(i); break }
}
if (!mode) failures.push('no sim/real mode control in the devices drawer — the spawn form did not render')

if (mode) {
  await mode.selectOption('real')
  await page.waitForTimeout(1200)
  const idField = page.locator('input[placeholder="lerobot id (optional)"]').first()
  if (!await idField.count()) failures.push('real mode renders no calibration id field')
  else {
    const verdictFor = async (typed) => {
      await idField.fill(typed)
      await page.waitForTimeout(600)
      const warn = (await page.locator('p.hint.warn').allInnerTexts()).join(' ').replace(/\s+/g, ' ')
      const ok = (await page.locator('p.hint.ok').allInnerTexts()).join(' ').replace(/\s+/g, ' ')
      return { warn, ok }
    }

    if (teleopOnly) {
      const v = await verdictFor(teleopOnly)
      notes.push(`note: "${teleopOnly}" -> ${(v.warn || v.ok || '(nothing)').slice(0, 200)}`)
      if (v.ok) failures.push(`"${teleopOnly}" is calibrated only as a teleoperator, and the form shows a GREEN verdict: ${v.ok.slice(0, 160)}`)
      if (!/teleoperator/i.test(v.warn)) failures.push(`"${teleopOnly}" does not say which side it was calibrated for: ${v.warn.slice(0, 160) || '(no warning at all)'}`)
      if (!/no calibration registered/i.test(v.warn)) failures.push('the warning omits the words lerobot itself will print')
      if (!/no joints/i.test(v.warn)) failures.push('the warning omits the symptom (presence, no joints), so the operator cannot connect the two')
    } else {
      notes.push('note: nothing on this machine is calibrated ONLY as a teleoperator, so the wrong-side case could not be exercised on the page')
    }

    if (robotSide) {
      const v = await verdictFor(robotSide)
      notes.push(`note: "${robotSide}" -> ${(v.ok || v.warn || '(nothing)').slice(0, 160)}`)
      // The check must not have become a blanket accusation: a real robot-side id stays green.
      if (!v.ok || v.warn) failures.push(`"${robotSide}" IS calibrated as a robot and the form does not confirm it: ${(v.warn || '(no verdict)').slice(0, 160)}`)
    }

    await idField.fill('')
  }
  await mode.selectOption('sim')  // leave the form as it was found
}

await browser.close()
for (const n of notes) console.log(n)
if (failures.length) {
  console.error(`FAIL (${failures.length})`)
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
console.log('PASS  the form never blesses a calibration from the wrong side of the pair')
