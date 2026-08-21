/**
 * Q88 — a client stuck in a refusal loop is VISIBLE ON THE PAGE, not only in curl.
 *
 * The server counts refused handshakes and explains them in /api/health (refusals.py), but until
 * this landed no frontend code read that endpoint at all: the news from a 19.3-hour storm existed
 * only for whoever thought to curl it. This audit proves the sentence reaches a screen, and — the
 * harder half — that it STAYS AWAY when it would be noise.
 *
 * Four specimens injected into /api/health, driving the real component:
 *   storm     -> the banner appears, names the client and the one fix, and tells the reader their
 *                own session is fine (they are signed in; the server cannot know that)
 *   handful   -> silence. A banner for someone signing in teaches the operator to ignore banners,
 *                and this dashboard's banners guard an e-stop.
 *   stopped   -> silence. "It stopped" is an answer in the payload, not an interruption.
 *   withheld  -> the unauthenticated shape (dd658b47: counts, no identities) still says the true,
 *                smaller thing, and invents no culprit.
 *
 * Reads only; no robot is commanded and the live mesh is untouched (only /api/health is faked).
 * Run: node scripts/audit-refusal-notice.mjs
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

const STORM = {
  total: 812, recent: 240, window_s: 300, clients: 1, storm: true,
  worst: { client: '192.168.1.44', path: '/ws/camera/so101-leader/main', kind: 'credential', count: 240 },
  text: '192.168.1.44 is retrying /ws/camera/so101-leader/main and being refused (credential) 240 times in the last 5 minutes, ~48/min. It will not recover by itself: that page is holding an expired or wrong sign-in - reload it and sign in again. Nothing is wrong with the robots.',
}
const SPECIMENS = {
  storm: STORM,
  handful: { total: 3, recent: 3, window_s: 300, clients: 1, storm: false, worst: { client: '10.0.0.5', path: '/ws/mesh', kind: 'credential', count: 3 }, text: '3 handshake(s) refused in the last 5 minutes.' },
  stopped: { total: 400, recent: 0, window_s: 300, clients: 0, storm: true, text: '400 handshake(s) refused since start, none in the last 5 minutes.' },
  withheld: { total: 40, recent: 40, window_s: 300, clients: 1, storm: true, text: '40 handshake(s) refused in the last 5 minutes. Sign in to see which client and which path.' },
}

let which = 'handful'
const browser = await chromium.launch()
// PWA: a service-worker response is not interceptable, and the fixture would silently never land.
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()
page.on('pageerror', e => check(false, 'page threw', e.message))

// Only /api/health is faked; everything else (mesh socket, fleet) stays real.
await page.route(/\/api\/health/, async route => {
  const real = await route.fetch()
  const body = await real.json().catch(() => ({ status: 'ok' }))
  await route.fulfill({ json: { ...body, refused_handshakes: SPECIMENS[which] } })
})

const banner = page.locator('.toast.warn', { hasText: 'being refused repeatedly' })
const show = async (name) => {
  which = name
  // The poll is 60s; a reload is the honest way to re-ask within an audit.
  await page.reload({ waitUntil: 'domcontentloaded' })
  await page.locator('.fleetbar').first().waitFor({ timeout: 20000 })
  await page.waitForTimeout(1500)
}

await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.locator('.fleetbar').first().waitFor({ timeout: 20000 })

await show('handful')
check(await banner.count() === 0, 'a handful of refusals is SILENT (someone signing in is not news)')

await show('stopped')
check(await banner.count() === 0, 'a storm that already stopped is SILENT (the payload answers "did my fix work?")')

await show('storm')
await banner.first().waitFor({ timeout: 10000 }).catch(() => {})
const text = (await banner.count()) ? (await banner.first().innerText()).replace(/\s+/g, ' ') : ''
check(/192\.168\.1\.44/.test(text), 'the loop is named on the page, with the client', text.slice(0, 120))
check(/\/ws\/camera\/so101-leader\/main/.test(text), 'and the path being refused')
check(/reload it and sign in again/.test(text), 'and the ONE action that ends it')
check(/Your own session is fine/.test(text), 'the reader is told it is not THEM — they are signed in')
check(/Nothing is wrong with the robots/.test(text), 'it exonerates the fleet (the whole point of Q88)')
check(!/undefined|NaN/.test(text), 'no placeholder leaked into the sentence')

await show('withheld')
const vague = (await banner.count()) ? (await banner.first().innerText()).replace(/\s+/g, ' ') : ''
check(/Something is being refused/.test(vague), 'the identity-withheld shape still says the true, smaller thing', vague.slice(0, 100))
check(!/192\.168|\/ws\//.test(vague), 'and invents no culprit it was never told about')

await page.screenshot({ path: '/tmp/refusal_notice_audit.png' })
await browser.close()
console.log(fails.length
  ? `  FAIL  ${fails.length} check(s): ${fails.join(' | ')}`
  : '  PASS  a refusal loop is named on the page, and the quiet cases stay quiet')
process.exit(fails.length ? 1 : 0)
