/**
 * Q88 — a sign-in that lapses WHILE THE TAB IS OPEN closes the gate again, and warns first.
 *
 * The measured incident this proves against: one phone spent 19.3 hours reopening camera sockets
 * that the server refused with 403, because its JWT expired mid-session and `AuthGate` decided
 * login-vs-open once on mount. The page looked alive and was deaf, so the operator would have
 * debugged the robot.
 *
 * The transition is driven through the REAL component, not a stub: the page is opened with the
 * working local token (so the gate opens for real), then `localStorage` is rewritten to a JWT with
 * an `exp` in the past — exactly what the phone was holding — and the tab is told it regained
 * focus, which is how a phone returning from the background re-checks. Nothing is mocked; the
 * assertion is what a person would see.
 *
 * Two specimens, because the pre-warning and the lockout are different products:
 *   expired-in-4h : `exp` 4 hours ago  -> the gate closes, and the sentence exonerates the robots
 *   lapses-in-3m  : `exp` 3 min ahead  -> the app STAYS open with an amber banner above it
 *
 * Reads only; no robot is commanded. The rewritten token is removed again at the end, and it never
 * leaves the browser context this script owns.
 * Run: node scripts/audit-session-lapse.mjs
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`,
  'utf8',
).trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'

const fails = []
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) fails.push(label)
}

/** An unsigned JWT with the given exp. It is never sent anywhere that verifies it. */
const jwt = (exp) => {
  const b64 = (o) => Buffer.from(JSON.stringify(o)).toString('base64')
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  return `${b64({ alg: 'HS256', typ: 'JWT' })}.${b64({ sub: 'audit', name: 'audit', exp })}.sIg`
}

const browser = await chromium.launch()
// PWA: a service-worker response is not interceptable, and an injected fixture silently never lands.
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()
page.on('pageerror', e => check(false, 'page threw', e.message))

await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.locator('.fleetbar').first().waitFor({ timeout: 20000 })
check(true, 'the dashboard opened with the working token (the gate is genuinely open)')
check(!(await page.locator('.sessionwarn').count()), 'a valid credential shows NO session banner')

// --- specimen 1: the lapse is 3 minutes away -> stay open, warn -----------------
await page.evaluate((t) => {
  localStorage.setItem('strands.token', t)
  window.dispatchEvent(new Event('focus'))
}, jwt(Math.floor(Date.now() / 1000) + 180))
const warn = page.locator('.sessionwarn')
await warn.waitFor({ timeout: 8000 }).catch(() => {})
const warnText = (await warn.count()) ? (await warn.first().innerText()).trim() : ''
check(/lapses in 3 minutes/.test(warnText), 'the banner says HOW LONG is left', JSON.stringify(warnText))
check(/recording/.test(warnText), 'it names the cost: a recording refused part-way through')
check(await page.locator('.fleetbar').count() > 0, 'the app is STILL USABLE while it warns (not gated early)')

// --- specimen 2: already lapsed -> the gate closes, with the reason -------------
await page.evaluate((t) => {
  localStorage.setItem('strands.token', t)
  window.dispatchEvent(new Event('focus'))
}, jwt(Math.floor(Date.now() / 1000) - 4 * 3600))
const gate = page.locator('.authgate')
await gate.waitFor({ timeout: 15000 })
const gateText = (await gate.innerText()).replace(/\s+/g, ' ').trim()
check(/expired 4 hours ago/.test(gateText), 'the gate says WHEN the sign-in lapsed', gateText.slice(0, 160))
check(/sign in again/i.test(gateText), 'it says what to do')
check(/Nothing is wrong with the robots/.test(gateText),
  'it exonerates the hardware — the whole point of Q88 (19.3h of hunting a camera bug)')
check(!(await page.locator('.fleetbar').count()), 'the stale, deaf dashboard is no longer on screen')

await page.screenshot({ path: '/tmp/session_lapse_audit.png' })
await page.evaluate(() => localStorage.removeItem('strands.token'))
await browser.close()
console.log(fails.length
  ? `  FAIL  ${fails.length} check(s): ${fails.join(' | ')}`
  : '  PASS  a mid-session lapse warns, then re-gates, and blames the session rather than the robot')
process.exit(fails.length ? 1 : 0)
