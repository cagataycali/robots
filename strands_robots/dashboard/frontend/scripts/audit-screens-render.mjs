/**
 * Every screen of the live dashboard RENDERS — no error boundary, no thrown exception.
 *
 * This exists because of Q54: the ⚙ devices screen crashed the instant it opened (React #310,
 * a useEffect below an early return) and the entire 600+ test suite could not see it, because
 * nothing in this repo renders a component in a test. A hook-order crash is only visible to a
 * real browser, and the crash card is polite enough that a human might not notice either.
 *
 * It clicks only NAVIGATION and read-only controls. Nothing here commands a robot: no spawn,
 * no twin, no record, no run, no teleop. Safe to run against the live rig with the arms on.
 *
 * Run: node scripts/audit-screens-render.mjs   (needs a running dashboard on :8090
 *      and node playwright — same requirements as audit-primary-actions.mjs)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN_FILE = process.env.STRANDS_DASH_TOKEN_FILE
  ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`
const TOKEN = fs.readFileSync(TOKEN_FILE, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const browser = await chromium.launch()
const page = await (await browser.newContext({
  viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
})).newPage()

const thrown = []
page.on('pageerror', e => thrown.push(`threw: ${(e.message || String(e)).slice(0, 200)}`))
page.on('console', m => {
  const t = m.text()
  // A 404 is the frontend asking for an endpoint this build's server half does not have yet
  // (e.g. /api/network/hint while a restart is pending) — the UI is expected to cope, and
  // Q21 made those answer JSON so it can. Anything else is a defect.
  if (m.type() === 'error' && !t.includes('404')) thrown.push(`console: ${t.slice(0, 200)}`)
})

const failures = []
const check = async (what) => {
  const cards = await page.evaluate(() =>
    [...document.querySelectorAll('.crashcard')].map(e => e.innerText.replace(/\n+/g, ' | ').slice(0, 160)))
  if (cards.length) { failures.push(`${what}: ${cards.join(' ; ')}`); console.log(`  CRASH ${what}\n          ${cards.join('\n          ')}`) }
  else console.log(`  ok    ${what}`)
}
const escape = async () => { await page.keyboard.press('Escape'); await page.waitForTimeout(600) }
const tap = async (selector, what) => {
  const el = page.locator(selector).first()
  if (!(await el.count())) { console.log(`  SKIP  ${what} (not on this rig)`); return false }
  await el.click({ timeout: 5000 }).catch(e => failures.push(`${what}: not clickable — ${String(e).slice(0, 80)}`))
  await page.waitForTimeout(2000)
  await check(what)
  return true
}

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(7000)
await check('fleet (first paint)')

for (const nav of ['devices', 'record', 'train', 'activity', 'settings', 'help']) {
  await escape()
  await tap(`button.chip:has-text("${nav}")`, `nav → ${nav}`)
}

// Settings has six tabs and each one mounts different fields.
await escape()
await tap('button.chip:has-text("settings")', 'settings (re-open)')
for (const tab of await page.evaluate(() => [...document.querySelectorAll('.tab, [role=tab]')].map(t => t.innerText.trim()))) {
  await tap(`.tab:has-text("${tab}"), [role=tab]:has-text("${tab}")`, `settings → ${tab}`)
}

// The robot detail screen, reached the way an operator reaches it: the robot's name.
await escape()
if (await tap('button.peername', 'robot detail (via the name)')) {
  await tap('button:has-text("cameras")', 'detail → cameras')
}

// Typing in the dataset search mounts the Hub result list, a different tree again.
await escape()
await tap('button.chip:has-text("train")', 'train (re-open)')
const search = page.locator('.train-sheet input[placeholder*="search"]').first()
if (await search.count()) {
  await search.fill('so101')
  await page.waitForTimeout(3500)
  await check('train → dataset search results')
}

// A11y rule this dashboard kept breaking: SELECTED state expressed in CSS only. `.chip.on` and
// `.tab.on` looked obvious on screen while a screen reader announced six identical "Mesh, button"
// tabs and two identical "wrist, button" camera switches — the selected one indistinguishable.
// Colour is not a state announcement, so a toggle that carries the marker class must also carry
// aria-pressed / aria-selected / aria-current.
const arialess = []
const iconOnly = []
const ICON_SCAN = () => page.evaluate(() => {
  const ICON_ONLY = /^[\p{Extended_Pictographic}\p{So}\p{Sk}\p{Sm}\s▶■✕▴▾↑↓⚙⚒☰⏺🛑🎙📷⚠?·+×-]*$/u
  return [...document.querySelectorAll('button')]
    .filter(b => ICON_ONLY.test((b.innerText || '').trim()))
    .filter(b => !b.getAttribute('aria-label') && !b.title && !b.getAttribute('aria-labelledby'))
    .map(b => `${b.className} "${(b.innerText || '').trim().slice(0, 8)}"`)
})
for (const [nav, extra] of [[null, null], ['settings', null], [null, 'peername'],
                            ['activity', null], ['devices', null], ['record', null],
                            ['train', null], ['help', null]]) {
  await escape()
  if (nav) await page.locator(`button.chip:has-text("${nav}")`).first().click().catch(() => {})
  if (extra) await page.locator(`button.${extra}`).first().click().catch(() => {})
  await page.waitForTimeout(1800)
  iconOnly.push(...await ICON_SCAN())
  // Q58: an overlay must own the keyboard while it is up, and hand it back on close. Before the
  // fix, five of six sheets left focus on the nav chip BEHIND them, so Tab walked the fleet the
  // sheet was covering. Also: a placeholder is not a label — it vanishes when the operator types.
  if (nav || extra) {
    const f = await page.evaluate(() => {
      const dlg = document.querySelector('[role=dialog], .drawer, .train-sheet, .detail')
      if (!dlg) return null
      const a = document.activeElement
      const unlabelled = [...dlg.querySelectorAll('input, select, textarea')]
        .filter(i => !i.getAttribute('aria-label') && !i.getAttribute('aria-labelledby') && !i.closest('label')
                     && !(i.id && dlg.querySelector(`label[for="${i.id}"]`)))
        .map(i => i.placeholder || i.tagName.toLowerCase())
      return { inside: !!(a && dlg.contains(a)), focus: a ? `${a.tagName.toLowerCase()}.${a.className}` : 'none', unlabelled }
    })
    if (f && !f.inside) { failures.push(`focus stayed outside ${nav ?? extra} (on ${f.focus})`); console.log(`  A11Y  opening ${nav ?? extra} left focus outside it, on ${f.focus}`) }
    for (const u of f?.unlabelled ?? []) { failures.push(`unlabelled field in ${nav ?? extra}: "${u}"`); console.log(`  A11Y  ${nav ?? extra}: form control with no label ("${u}")`) }
  }
  arialess.push(...await page.evaluate(() => [...document.querySelectorAll('button')]
    .filter(b => /(^|\s)(on|active|selected)(\s|$)/.test(b.className))
    .filter(b => !b.getAttribute('aria-pressed') && !b.getAttribute('aria-selected') && !b.getAttribute('aria-current'))
    .map(b => `${b.className} "${b.innerText.trim().slice(0, 20)}"`)))
}
// Second a11y rule, same family: a button whose whole label is an ICON ("↑", "✕", "⚙") with no
// aria-label or title is announced as a bare "button". The agent dock's send arrow and four of the
// five sheet-closing ✕s were exactly that, while HelpSheet's ✕ had been labelled all along — so it
// was drift, not a decision. Scanned per screen, because a sheet's buttons only exist while it is open.
const uniqueArialess = [...new Set(arialess)]
if (uniqueArialess.length) {
  for (const a of uniqueArialess) { failures.push(`selected-but-silent: ${a}`); console.log(`  A11Y  selected state is CSS-only: ${a}`) }
} else {
  console.log('  ok    every selected toggle announces itself (aria-pressed/selected/current)')
}
if (iconOnly.length) {
  for (const i of [...new Set(iconOnly)]) { failures.push(`icon-only, unnamed: ${i}`); console.log(`  A11Y  icon button with no accessible name: ${i}`) }
} else {
  console.log('  ok    every icon-only button has an accessible name')
}

await browser.close()
for (const t of thrown) { failures.push(t); console.log(`  ERROR ${t}`) }
if (failures.length) {
  console.log(`  FAIL  ${failures.length} problem(s) while rendering the dashboard:`)
  for (const f of failures) console.log(`          ${f}`)
  process.exit(1)
}
console.log('  PASS  every screen renders, nothing thrown')
