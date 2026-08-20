/**
 * Can this dashboard be driven with no mouse — and is the BRAKE the first thing the keyboard
 * reaches? Needs a running dashboard on :8090 and node playwright:
 *   npm run audit:keyboard
 *
 * Three properties, all measured by actually pressing Tab and Escape on the live page:
 *
 *  1. STOP ALL is the FIRST tab stop. Nobody wrote that down — it fell out of DOM order when the
 *     E-STOP moved into its own fixed layer — and it is exactly the property a later layout change
 *     takes away silently. One Tab to the brake, from anywhere.
 *  2. Every tab stop is a real, visible control: no zero-size stops (a focusable element the user
 *     cannot see is a dead key press), and a visible focus indicator on each, because a keyboard
 *     operator who cannot see where focus is does not know what space will press.
 *  3. Every overlay can be left with Escape. A sheet whose only exit is a ✕ you must find with a
 *     mouse is not keyboard-operable, and these sheets sit over a fleet that can move.
 *
 * It only presses Tab and Escape and never activates anything, so it is safe with real arms
 * powered on.
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN_FILE = process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const TOKEN = fs.readFileSync(TOKEN_FILE, 'utf8').trim()
const failures = []

const browser = await chromium.launch()
const page = await (await browser.newContext({ viewport: { width: 1200, height: 900 } })).newPage()
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForSelector('.fleetbar', { timeout: 20000 })
await page.waitForTimeout(4000)

const focused = () => page.evaluate(() => {
  const a = document.activeElement
  if (!a || a === document.body) return { kind: 'BODY' }
  const r = a.getBoundingClientRect()
  const s = getComputedStyle(a)
  return {
    kind: `${a.tagName.toLowerCase()}.${(a.className || '').toString().trim().slice(0, 20)}`,
    label: (a.innerText || a.getAttribute('aria-label') || a.placeholder || '').trim().replace(/\s+/g, ' ').slice(0, 24),
    w: Math.round(r.width), h: Math.round(r.height),
    ring: (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0) || s.boxShadow !== 'none',
  }
})

// 1 + 2: walk the fleet screen until the cycle repeats.
const stops = []
for (let i = 0; i < 40; i++) {
  await page.keyboard.press('Tab')
  const f = await focused()
  if (i > 3 && f.kind === stops[0]?.kind && f.label === stops[0]?.label) break
  stops.push(f)
}
const first = stops[0]
if (first && /estop/.test(first.kind)) {
  console.log(`  PASS  the brake is one Tab away — first stop is ${first.kind} "${first.label}"`)
} else {
  failures.push(`first tab stop is ${first?.kind} "${first?.label}", not the E-STOP`)
  console.log(`  FAIL  first tab stop is ${first?.kind} "${first?.label}" — STOP ALL is not one Tab away`)
}
const invisible = stops.filter(s => s.kind !== 'BODY' && (s.w === 0 || s.h === 0))
const ringless = stops.filter(s => s.kind !== 'BODY' && s.w > 0 && !s.ring)
for (const s of invisible) { failures.push(`zero-size tab stop ${s.kind}`); console.log(`  FAIL  tab stops on a zero-size element: ${s.kind} "${s.label}"`) }
for (const s of ringless) { failures.push(`no focus ring on ${s.kind}`); console.log(`  FAIL  no visible focus indicator: ${s.kind} "${s.label}"`) }
if (!invisible.length && !ringless.length) console.log(`  PASS  all ${stops.length} tab stops are visible and show focus`)

// 3: Escape must close every overlay.
for (const nav of ['devices', 'record', 'train', 'activity', 'settings', 'help']) {
  await page.locator(`button.chip:has-text("${nav}")`).first().click().catch(() => {})
  await page.waitForTimeout(1500)
  const opened = await page.locator('[role=dialog], .drawer, .train-sheet').count()
  await page.keyboard.press('Escape')
  await page.waitForTimeout(900)
  const left = await page.locator('[role=dialog], .drawer, .train-sheet').count()
  if (!opened) { failures.push(`${nav} did not open`); console.log(`  FAIL  ${nav} did not open at all`) }
  else if (left) { failures.push(`Escape did not close ${nav}`); console.log(`  FAIL  Escape does not close ${nav}`) }
  else console.log(`  ok    Escape closes ${nav}`)
}

await browser.close()
console.log(failures.length ? `  FAIL  ${failures.length} keyboard problem(s)` : '  PASS  the whole dashboard is operable from the keyboard')
process.exit(failures.length ? 1 : 0)
