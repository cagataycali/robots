/**
 * Q133: the full-bleed screens in LANDSCAPE — 844x390.
 *
 * `.train-sheet` (record + training) is `position: fixed; inset: 0`, so it COVERS the app: the only
 * way out is its own ✕, and the only way to start work is a button near the bottom of a long form.
 * Rotate a phone and the height budget collapses to 390px. Two distinct ways that breaks, both of
 * them dead ends for the operator rather than cosmetic:
 *   - the primary action cannot be reached or hit → the work cannot be started;
 *   - the ✕ cannot be reached → the screen cannot be LEFT, on a layer that hides everything else.
 *
 * Q132 proved this contract for `.sheet` (the centred modals) against the stylesheet. These two are a
 * different geometry — full-bleed, its own scroll container — and they are the screens where the real
 * work happens, so this measures the LIVE app rather than a harness.
 *
 * Read-only: it opens two screens, scrolls, and measures. It never fills the form or presses start.
 */
import fs from 'node:fs'
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const LAND = { width: 844, height: 390 }

const failures = []
const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: LAND, isMobile: true, hasTouch: true, serviceWorkers: 'block' })
const page = await ctx.newPage()
page.on('pageerror', e => failures.push(`page threw: ${String(e.message).slice(0, 140)}`))
await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)

for (const screen of ['record', 'train']) {
  const chip = page.locator(`button.chip:has-text("${screen}")`).first()
  if (!(await chip.count())) { failures.push(`${screen}: no chip to open it at ${LAND.width}x${LAND.height}`); continue }
  await chip.click()
  await page.waitForTimeout(2500)
  const sheet = page.locator('.train-sheet').first()
  if (!(await sheet.count())) { failures.push(`${screen}: the full-bleed sheet did not render`); continue }

  const m = await page.evaluate(() => {
    const s = document.querySelector('.train-sheet')
    const cs = getComputedStyle(s)
    // The primary action is the .go button inside the form (scoped: `.go` alone also matches the
    // agent chat's ▶ Run — a earlier audit spent its clicks on that one).
    const go = s.querySelector('.train-form button.go') ?? s.querySelector('button.go')
    const close = s.querySelector('.train-head .dock-min')
    const hitAt = el => {
      if (!el) return null
      const r = el.getBoundingClientRect()
      const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
      return { inView: r.top >= 0 && r.bottom <= window.innerHeight + 1, hits: at === el || el.contains(at),
        who: at?.className?.toString?.().slice(0, 40) ?? null }
    }
    // Reach the action the way a human does: scroll UNTIL IT IS VISIBLE. Not "scroll to the
    // bottom" — that was my wrong assumption and it reported the train screen's ▶ train button as
    // unreachable: that form sits near the TOP with long checkpoint/dataset lists below it, so
    // scrolling to the bottom scrolls the action back OFF the screen.
    go?.scrollIntoView({ block: 'center' })
    const action = hitAt(go)
    // Then reach the exit the way a human does: scroll back to the top.
    s.scrollTop = 0
    const exit = hitAt(close)
    return {
      overflowY: cs.overflowY,
      scrollable: s.scrollHeight > s.clientHeight + 1,
      sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      hasGo: !!go, hasClose: !!close, action, exit,
      goText: go?.innerText?.trim().slice(0, 30) ?? null,
    }
  })

  if (m.sideways) failures.push(`${screen}: the page scrolls SIDEWAYS in landscape`)
  if (m.scrollable && !/auto|scroll/.test(m.overflowY)) {
    failures.push(`${screen}: content overflows but computed overflow-y is "${m.overflowY}" — a full-bleed `
      + 'layer that clips its own content has no way back')
  }
  if (!m.hasGo) failures.push(`${screen}: no primary action button found in the sheet`)
  else if (!m.action.inView || !m.action.hits) {
    failures.push(`${screen}: the primary action ("${m.goText}") is not reachable at any scroll position (`
      + `scrolled into view, then measured (inView ${m.action.inView}, hit-testable ${m.action.hits}${m.action.who ? `, point belongs to "${m.action.who}"` : ''})`)
  }
  if (!m.hasClose) failures.push(`${screen}: no ✕ in the head — a full-bleed layer with no exit`)
  else if (!m.exit.inView || !m.exit.hits) {
    failures.push(`${screen}: the ✕ is not reachable after scrolling back to the top `
      + `(inView ${m.exit.inView}, hit-testable ${m.exit.hits}) — the operator cannot LEAVE this screen`)
  }
  if (m.hasGo && m.hasClose && m.action?.hits && m.exit?.hits) {
    console.log(`${screen}: scrollable=${m.scrollable}, action "${m.goText}" reachable, ✕ reachable at ${LAND.width}x${LAND.height}`)
  }
  await page.keyboard.press('Escape').catch(() => {})
  await page.waitForTimeout(600)
}

await browser.close()
if (failures.length) { console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n')); process.exit(1) }
console.log('landscape actions: on a rotated phone both full-bleed screens scroll, their primary action can be '
  + 'reached AND hit, and their ✕ comes back — the screen can be both started and left')
