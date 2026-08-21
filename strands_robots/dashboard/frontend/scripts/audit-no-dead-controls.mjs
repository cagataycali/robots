/**
 * Q134: no visible control on any screen may sit under an OVERLAY LAYER.
 *
 * Q133 was the case: the record screen's ✕ lay under `.estop-layer` (fixed, z-index 200), so on every
 * phone the gesture for "close this panel" pressed the fleet's emergency stop, and the panel could not
 * be closed at all. That bug was invisible to every desktop audit — the collision only exists below
 * ~944px — and invisible to reading the CSS, because both elements are correct in isolation. The only
 * thing that finds it is asking the page WHO OWNS THE POINT a finger lands on.
 *
 * This sweeps every screen at 390x844 for that class. Two scoping rules, both learned by getting 23
 * false positives first, and both essential — an audit that cries wolf gets muted:
 *
 *  1. SCOPE TO THE TOPMOST LAYER. An open panel is SUPPOSED to cover the chips beneath it; scanning
 *     the whole document reported the overlay working correctly as 19 defects.
 *  2. THE OWNER MUST BE IN A DIFFERENT FIXED/STICKY LAYER. Four more candidates were same-box
 *     wrappers and duplicate bars: elementFromPoint named a neighbour, but a REAL mouse click at that
 *     exact point focused the agent input and typed into it (verified by keystroke), so nothing was
 *     dead. A same-box overlay forwards the click; a different layer eats it. Only the latter is the
 *     bug, and it is exactly what Q133 was.
 *
 * Read-only: it opens screens, scrolls, and measures. It clicks only chips to navigate.
 */
import fs from 'node:fs'
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const VP = { width: 390, height: 844 }

const failures = []
const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: VP, isMobile: true, hasTouch: true, serviceWorkers: 'block' })
const page = await ctx.newPage()
page.on('pageerror', e => failures.push(`page threw: ${String(e.message).slice(0, 140)}`))
await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)

const scan = () => page.evaluate(() => {
  const layers = [...document.querySelectorAll('.sheet-backdrop, .train-sheet, .drawer')]
  const root = layers.length ? layers[layers.length - 1] : document
  const fixedLayer = el => {
    for (let n = el; n && n !== document.body; n = n.parentElement) {
      const cs = getComputedStyle(n)
      if (cs.position === 'fixed' || cs.position === 'sticky') return n
    }
    return null
  }
  const out = []
  for (const el of root.querySelectorAll('button, a[href], input, select, textarea, [role="button"]')) {
    const r = el.getBoundingClientRect()
    if (r.width < 4 || r.height < 4 || r.bottom <= 0 || r.top >= innerHeight) continue
    const cs = getComputedStyle(el)
    if (cs.visibility === 'hidden' || cs.pointerEvents === 'none' || cs.opacity === '0') continue
    const x = Math.min(Math.max(r.x + r.width / 2, 1), innerWidth - 1)
    const y = Math.min(Math.max(r.y + r.height / 2, 1), innerHeight - 1)
    const at = document.elementFromPoint(x, y)
    if (!at || at === el || el.contains(at) || at.contains(el)) continue
    // Rule 2: only a DIFFERENT fixed/sticky layer eats a tap.
    const mine = fixedLayer(el), theirs = fixedLayer(at)
    if (!theirs || theirs === mine) continue
    // Rule 3, the one that separates "dead" from "currently scrolled under a bar": bring the control
    // to the MIDDLE of the viewport and ask again. A card control that sits beneath the sticky header
    // or the bottom dock moves out from under it — the user scrolls, which is not a defect. Q133's ✕
    // lived in a FIXED head, so it could not move and stayed under the e-stop at every scroll
    // position. That is what dead means.
    el.scrollIntoView({ block: 'center' })
    const r2 = el.getBoundingClientRect()
    const at2 = document.elementFromPoint(
      Math.min(Math.max(r2.x + r2.width / 2, 1), innerWidth - 1),
      Math.min(Math.max(r2.y + r2.height / 2, 1), innerHeight - 1))
    if (!at2 || at2 === el || el.contains(at2) || at2.contains(el)) continue
    const theirs2 = fixedLayer(at2)
    if (!theirs2 || theirs2 === mine) continue
    const ar = at2.getBoundingClientRect()
    // A same-box overlay forwards the click (measured: the agent input took a real keystroke through one).
    if (Math.abs(ar.x - r.x) < 2 && Math.abs(ar.y - r.y) < 2 && Math.abs(ar.width - r.width) < 2) continue
    out.push({
      control: `${el.tagName.toLowerCase()}.${(el.className?.toString?.() || '').slice(0, 24)}`,
      label: (el.innerText || el.getAttribute('aria-label') || el.placeholder || '').trim().slice(0, 30),
      owner: (theirs2.className?.toString?.() || theirs2.tagName).slice(0, 30),
      z: getComputedStyle(theirs2).zIndex,
    })
  }
  return out
})

for (const [name, chip] of [['fleet', null], ['record', 'record'], ['train', 'train'],
  ['devices', 'devices'], ['settings', 'settings'], ['activity', 'activity']]) {
  if (chip) {
    const c = page.locator(`button.chip:has-text("${chip}")`).first()
    if (!(await c.count())) { failures.push(`${name}: no chip opens it at ${VP.width}px`); continue }
    await c.click()
    await page.waitForTimeout(2200)
  }
  // Both scroll positions: Q133's ✕ was only under the e-stop at the top, and a bottom dock only
  // covers content at the end of the scroll.
  let found = []
  for (const pos of ['top', 'bottom']) {
    await page.evaluate(p => {
      const layers = [...document.querySelectorAll('.sheet-backdrop, .train-sheet, .drawer')]
      const s = layers.length ? layers[layers.length - 1] : document.scrollingElement
      s.scrollTop = p === 'bottom' ? s.scrollHeight : 0
    }, pos)
    await page.waitForTimeout(500)
    found = found.concat((await scan()).map(h => ({ ...h, pos })))
  }
  for (const h of found) {
    failures.push(`${name} (${h.pos}): ${h.control} "${h.label}" is under "${h.owner}" (z ${h.z}) — `
      + 'a tap there hits the layer, not the control')
  }
  console.log(`${name}: ${found.length ? found.length + ' DEAD' : 'no dead controls'}`)
  if (chip) { await page.keyboard.press('Escape').catch(() => {}); await page.waitForTimeout(700) }
}

await browser.close()
if (failures.length) { console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n')); process.exit(1) }
console.log(`no dead controls: on all six screens at ${VP.width}x${VP.height}, every visible control owns the point `
  + 'a finger lands on — nothing hides under the e-stop or any other fixed layer')
