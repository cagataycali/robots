/**
 * Q132: a modal's ACTION must be reachable on a short viewport.
 *
 * Five components render a `.sheet` (Estop, CameraConfig, Help, RunConfirm, Consent) and every one
 * of them puts the decision at the BOTTOM, in `.sheet-actions`. That is the classic modal defect: a
 * sheet taller than the screen whose confirm button sits below the fold, in a dialog that by design
 * covers everything else. On a phone in LANDSCAPE the budget is brutal — 86dvh of a 390px-tall
 * window is ~335px — and an operator who cannot reach "cancel" cannot get out of the dialog at all.
 *
 * This is a CSS-CONTRACT audit, not a component one: it builds a deliberately over-tall sheet from
 * the SHIPPED stylesheet and measures the layout every sheet inherits. That way it covers all five
 * (and any sheet added later) without five sets of fixtures that each drift on their own, and it
 * needs no live server — so it also runs when the rig is off.
 *
 * What it proves, in the order the failures matter:
 *   1. the harness is actually testing the hard case (the action starts BELOW the fold);
 *   2. the sheet fits inside the viewport rather than growing past it;
 *   3. it is genuinely scrollable (computed overflow-y, not just an intention in the CSS);
 *   4. after scrolling, the action's centre passes elementFromPoint — hit-testable, not merely
 *      present in the DOM. A button under the backdrop reads as "the click does nothing".
 */
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'

const dist = 'dist/assets'
const cssFile = fs.readdirSync(dist).filter(f => f.endsWith('.css')).sort()[0]
if (!cssFile) { console.error('FAIL\n - no built stylesheet in dist/assets: run `npm run build` first'); process.exit(1) }
const css = fs.readFileSync(path.join(dist, cssFile), 'utf8')

const failures = []
const browser = await chromium.launch()

// Portrait AND landscape: landscape is where 86dvh gets tight, and a phone rotates without asking.
for (const vp of [{ width: 390, height: 844, name: 'portrait' }, { width: 844, height: 390, name: 'landscape' }]) {
  const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height } })
  const page = await ctx.newPage()
  await page.setContent(`<!doctype html><html><body><div id="root"></div></body></html>`)
  await page.addStyleTag({ content: css })
  await page.evaluate(() => {
    // A real sheet's shape: heading, a lot of prose (the consent sheets explain a security grant),
    // then the decision row last.
    const long = Array.from({ length: 24 }, (_, i) =>
      `<p>Paragraph ${i + 1}: this sheet explains what is about to happen and why it needs a decision.</p>`).join('')
    document.body.innerHTML = `<div class="sheet-backdrop"><div class="sheet">
      <h2>Allow this?</h2>${long}
      <div class="sheet-actions">
        <button class="btn ghost" id="cancel">cancel</button>
        <button class="btn go" id="confirm">allow and respawn</button>
      </div></div></div>`
  })
  const m = await page.evaluate(() => {
    const sheet = document.querySelector('.sheet')
    const confirm = document.getElementById('confirm')
    const s = sheet.getBoundingClientRect()
    const cs = getComputedStyle(sheet)
    return {
      sheetBottom: s.bottom, sheetTop: s.top, sheetH: s.height,
      overflowY: cs.overflowY,
      scrollable: sheet.scrollHeight > sheet.clientHeight + 1,
      actionBelowFold: confirm.getBoundingClientRect().top > window.innerHeight,
      vh: window.innerHeight,
    }
  })
  if (!m.actionBelowFold && !m.scrollable) {
    failures.push(`${vp.name}: the harness is not testing the hard case — the action already fits, `
      + 'so this audit would pass on a sheet that cannot scroll. Add more content.')
  }
  if (m.sheetBottom > m.vh + 1 || m.sheetTop < -1) {
    failures.push(`${vp.name}: the sheet grows OUTSIDE the viewport `
      + `(top ${Math.round(m.sheetTop)}, bottom ${Math.round(m.sheetBottom)}, vh ${m.vh}) — `
      + 'content past the edge of a dialog that covers the page is unreachable by any gesture')
  }
  if (!/auto|scroll/.test(m.overflowY)) {
    failures.push(`${vp.name}: the sheet's computed overflow-y is "${m.overflowY}" — an over-tall `
      + 'sheet then clips its own decision row')
  }
  // Now do what a human does: scroll the sheet, then check the button can actually be HIT.
  const hit = await page.evaluate(() => {
    const sheet = document.querySelector('.sheet')
    sheet.scrollTop = sheet.scrollHeight
    const b = document.getElementById('confirm')
    const r = b.getBoundingClientRect()
    const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
    return {
      inView: r.top >= 0 && r.bottom <= window.innerHeight + 1,
      hitId: at?.id ?? at?.className ?? null,
      hits: at === b || b.contains(at),
    }
  })
  if (!hit.inView) failures.push(`${vp.name}: after scrolling to the bottom the confirm button is still off-screen`)
  if (!hit.hits) failures.push(`${vp.name}: the confirm button is not hit-testable — the point at its centre belongs to "${hit.hitId}"`)
  if (!failures.length) console.log(`${vp.name} ${vp.width}x${vp.height}: sheet ${Math.round(m.sheetH)}px inside ${m.vh}px, scrollable, confirm reachable and hit-testable`)
  await ctx.close()
}

await browser.close()
if (failures.length) {
  console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n'))
  process.exit(1)
}
console.log('sheet reachable: every .sheet stays inside the viewport, scrolls, and its decision row can be '
  + 'reached AND hit at 390x844 and 844x390')
