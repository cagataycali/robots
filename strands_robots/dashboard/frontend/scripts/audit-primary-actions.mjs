/**
 * Measures whether an ENABLED action on the live dashboard can be mistaken for a
 * disabled one — UX_REVIEW #2 ("stop dressing primary actions as disabled").
 *
 * Needs a running dashboard on :8090 and node playwright:
 *   node scripts/audit-primary-actions.mjs
 *
 * A button is SUSPECT when it is enabled, its text colour is the app's disabled
 * grey (--dim #8b98a9) and it has no fill. Navigation chips and tabs are grey on
 * purpose — they are not actions — so read the report, do not treat it as a gate.
 * Baseline at the time of writing (2026-08-19, after this commit): 0 suspect
 * BUTTONS, the only greys left being .chip / .tab navigation.
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'
const token = fs.readFileSync('/Users/cagatay/.strands_dashboard/local_api_token.txt','utf8').trim()
const PRIMARY = /^(run|▶|send|↑|start|save|create|train|enroll|apply|confirm|record|⏺|launch|spawn|add|connect|sign in|calibrate|deploy|submit|measure|probe|rescan|stop|■|🛑)/i
const b = await chromium.launch()
const ctx = await b.newContext({ serviceWorkers: 'block', viewport: { width: 1440, height: 950 } })
const page = await ctx.newPage()
await page.addInitScript(t => localStorage.setItem('strands.token', t), token)
await page.goto('http://localhost:8090/', { waitUntil: 'domcontentloaded' })
await page.waitForSelector('.fleetbar', { timeout: 20000 })
await page.waitForTimeout(2000)

const audit = () => page.evaluate((src) => {
  const PRIMARY = new RegExp(src, 'i')
  const out = []
  for (const el of document.querySelectorAll('button')) {
    const t = el.textContent.trim()
    if (!t || !PRIMARY.test(t)) continue
    const cs = getComputedStyle(el)
    const r = el.getBoundingClientRect()
    if (r.width === 0) continue
    const bg = cs.backgroundColor
    const transparentish = bg === 'rgba(0, 0, 0, 0)' || /rgba\(.*0(\.\d+)?\)$/.test(bg) ||
      (() => { const m = bg.match(/[\d.]+/g); return m && m.length === 4 && parseFloat(m[3]) < 0.5 })()
    const greyText = /^rgb\(1[23]\d, 1[45]\d, 1[56]\d\)$/.test(cs.color)
    out.push({ text: t.slice(0, 20), cls: el.className, disabled: el.disabled,
      color: cs.color, bg, opacity: cs.opacity,
      suspect: !el.disabled && greyText && (transparentish || bg === 'rgb(22, 29, 41)') })
  }
  return out
}, PRIMARY.source)

const screens = [
  ['home', null],
  ['devices', 'chip:has-text("devices")'],
  ['record', 'chip:has-text("record")'],
  ['train', 'chip:has-text("train")'],
  ['activity', 'chip:has-text("activity")'],
  ['settings', 'chip:has-text("settings")'],
]
const suspects = []
for (const [name, sel] of screens) {
  if (sel) {
    await page.keyboard.press('Escape').catch(() => {})
    await page.locator('.' + sel).first().click().catch(() => {})
    await page.waitForTimeout(1200)
  }
  const rows = await audit()
  const bad = rows.filter(r => r.suspect)
  console.log(`${name}: ${rows.length} primary-ish buttons, ${bad.length} suspect`)
  bad.forEach(r => { console.log('   SUSPECT', JSON.stringify(r)); suspects.push([name, r]) })
}
console.log('TOTAL SUSPECTS:', suspects.length)
await b.close()
