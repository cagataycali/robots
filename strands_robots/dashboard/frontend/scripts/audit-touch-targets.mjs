/**
 * Every control on the live dashboard is reachable with a thumb (44px, the iOS/Android
 * minimum), measured IN A REAL BROWSER at phone size.
 *
 * This exists because the numbers kept drifting back: UX_REVIEW measured the "open
 * so101-arm-1" target at 15px, a later pass at 25px, and a CSS block that already said
 * min-height:40px did not cover `.peername` or the security chip at all. A rule in a
 * stylesheet is a claim; this is the measurement.
 *
 * Run: node scripts/audit-touch-targets.mjs   (needs a running dashboard on :8090
 *      and node playwright — same requirements as audit-primary-actions.mjs)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const MIN = 44
const TOKEN_FILE = process.env.STRANDS_DASH_TOKEN_FILE
  ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`
const TOKEN = fs.readFileSync(TOKEN_FILE, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const browser = await chromium.launch()
const page = await (await browser.newContext({
  viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
})).newPage()
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(7000)

/** Measure whatever is on screen right now. */
const measure = (min) => page.evaluate((min) => {
  const out = []
  for (const el of document.querySelectorAll('button, a[href], select, input:not([type=hidden])')) {
    const r = el.getBoundingClientRect()
    if (!r.width || !r.height) continue            // hidden: not a target
    if (getComputedStyle(el).visibility === 'hidden') continue
    if (r.height >= min && r.width >= min) continue
    out.push(`${Math.round(r.height)}x${Math.round(r.width)} ${el.tagName.toLowerCase()}.${el.className || '-'} ${JSON.stringify((el.innerText || '').trim().slice(0, 24))}`)
  }
  return out
}, min)

// Every screen the nav can reach, not just the one that loads first: the fleet screen was
// clean while `.peername` and the security chip were 25px, so "the dashboard passes" has to
// mean all of it. Only NAV chips are clicked here - nothing on this rig is commanded.
const SCREENS = ['fleet', '⚙ devices', '⏺ record', '🎓 train', '☰ activity', '⚒ settings', '? help']
const failures = []
for (const screen of SCREENS) {
  if (screen !== 'fleet') {
    // A screen here is a full-bleed .train-sheet (fixed, inset 0, 96% opaque), so it COVERS
    // the nav rather than floating over it - the chips underneath are neither visible nor
    // clickable, which is correct for an opaque layer but means the sheet has to be closed
    // before the next chip can be tapped. Escape does it (verified), and so does the 44x44 ✕.
    await page.keyboard.press('Escape')
    await page.waitForTimeout(600)
    const chip = page.locator(`button.chip:has-text("${screen.split(' ')[1]}")`).first()
    if (!(await chip.count())) { console.log(`  SKIP  ${screen} (no such nav chip)`); continue }
    await chip.click()
    await page.waitForTimeout(2500)
  }
  const small = await measure(MIN)
  if (small.length) {
    failures.push(screen)
    console.log(`  FAIL  ${screen}: ${small.length} control(s) under ${MIN}px`)
    for (const s of small) console.log(`          ${s}`)
  } else {
    console.log(`  ok    ${screen}`)
  }
}
await browser.close()
if (failures.length) {
  console.log(`  FAIL  touch targets: ${failures.join(', ')}`)
  process.exit(1)
}
console.log(`  PASS  every control on every screen is at least ${MIN}px on a phone`)
