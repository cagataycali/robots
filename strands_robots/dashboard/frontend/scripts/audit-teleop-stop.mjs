/**
 * U22: a teleop stream can be SEEN and STOPPED from the screen — and a stop is only called stopped
 * when the arm says so (browser-proven).
 *
 * NOTHING REACHES HARDWARE — and that claim is now PROVEN rather than asserted. The first run of this
 * audit was written with the glob '**\/api/robots/*\/teleop*', and playwright's '*' does not cross a
 * '/', so it matched the GET and MISSED '/teleop/stop' entirely: two real stop commands went to a real
 * arm (harmless in that they only remove commands, and nothing was streaming — but unintended, which is
 * the point). The lesson is the Q30 law again: a test that cannot prove its own isolation must refuse
 * to run. So every teleop request is counted by the interceptor AND by a page-level listener, and any
 * difference FAILS the audit as an escape rather than being discovered later in a log.
 *
 * What only a browser can answer: does the stop affordance appear ONLY when frames are on the wire, is
 * it two-step (a mis-click during a good take costs the operator the recording), and — the assertion
 * this arc exists for — does the page REFUSE to claim success when the re-ask still shows a stream?
 * lib/teleopView.test.mjs pins that rule; this pins that the rule is what the screen obeys.
 *
 * Run: node scripts/audit-teleop-stop.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const following = { health: { worst: { state: 'following', headline: 'following so101-leader at 9.8Hz' }, publishers: {}, receivers: {} } }
const idle = { health: { receivers: {}, publishers: {} } }

const browser = await chromium.launch()
// A PWA serves from a service worker, which page.route CANNOT intercept — an injected fixture would
// silently never land and this audit would blame the UI for a fixture that never arrived.
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 1000 } })
const page = await ctx.newPage()

let stopPosts = 0
let statusAnswer = following   // flipped after the stop is accepted
// Route order: playwright matches in REVERSE registration order, so the SPECIFIC pattern goes LAST.
let seen = 0, handled = 0        // the tripwire: these must stay equal
page.on('request', r => { if (/\/api\/robots\/[^/]+\/teleop(\/|$)/.test(r.url())) seen += 1 })
await page.route(/\/api\/robots\/[^/]+\/teleop(\/.*)?$/, async route => {
  handled += 1
  const req = route.request()
  if (req.method() === 'POST') { stopPosts += 1; statusAnswer = idle
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ result: { ok: true } }) }) }
  return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(statusAnswer) })
})

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForResponse(r => r.url().includes('/api/fleet'), { timeout: 20_000 }).catch(() => {})
await page.waitForTimeout(1800)
const anyRobot = page.locator('text=/so101/').first()
if (await anyRobot.count() === 0) {
  console.log('  SKIP  no robot on this fleet to open — nothing to audit')
  await browser.close(); process.exit(0)
}
await anyRobot.click()
await page.waitForTimeout(1000)

const teleopBtn = page.locator('button', { hasText: /^teleop$/ }).first()
if (await teleopBtn.count() === 0) { failures.push('the robot detail view has no teleop button at all') }
else {
  await teleopBtn.click()
  await page.waitForTimeout(1200)
  const body = page.locator('div.hint').filter({ hasText: 'teleop:' }).first()
  const text = (await body.innerText().catch(() => '')).replace(/\s+/g, ' ')
  if (!/following so101-leader at 9\.8Hz/.test(text)) failures.push(`the streaming verdict did not reach the screen: ${text.slice(0, 120)}`)
  if (!/frames are on the wire/.test(text)) failures.push('a live stream is not announced as frames on the wire')

  // ONE CLICK MUST NOT STOP AN ARM. The first click only arms it.
  const arm = page.locator('button', { hasText: /^stop teleop$/ }).first()
  if (await arm.count() === 0) failures.push('frames are on the wire and there is no way to stop them from this screen')
  else {
    await arm.click(); await page.waitForTimeout(400)
    if (stopPosts !== 0) failures.push('the FIRST click stopped teleop — a mis-click during a good take must cost nothing')
    const cancel = page.locator('button', { hasText: /keep it running/ }).first()
    if (await cancel.count() === 0) failures.push('the armed state offers no way back — an operator who mis-clicked is trapped into stopping')
    const confirm = page.locator('button', { hasText: /^confirm — stop teleop/ }).first()
    if (await confirm.count() === 0) failures.push('no confirm affordance after arming')
    else {
      await confirm.click()
      await page.waitForTimeout(1800)
      if (stopPosts !== 1) failures.push(`confirm sent ${stopPosts} stop request(s), expected exactly 1`)
      const after = (await page.locator('div.hint').filter({ hasText: 'teleop' }).first().innerText().catch(() => '')).replace(/\s+/g, ' ')
      if (!/teleop stopped/.test(after)) failures.push(`the measured result did not reach the screen: ${after.slice(0, 140)}`)
      console.log(`  note  after the confirmed stop the page says: ${after.slice(0, 130)}`)
    }
  }
}

// THE ASSERTION THIS ARC EXISTS FOR: a stop whose re-ask STILL shows a stream must not read as success.
// The interception makes the re-ask answer "following" again, so the page is asked to lie and must not.
stopPosts = 0
const page2 = await ctx.newPage()
page2.on('request', r => { if (/\/api\/robots\/[^/]+\/teleop(\/|$)/.test(r.url())) seen += 1 })
await page2.route(/\/api\/robots\/[^/]+\/teleop(\/.*)?$/, async route => (handled += 1,
  route.request().method() === 'POST'
    ? (stopPosts += 1, route.fulfill({ status: 200, contentType: 'application/json', body: '{"result":{"ok":true}}' }))
    : route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(following) })))
await page2.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page2.waitForTimeout(2500)
await page2.locator('text=/so101/').first().click()
await page2.waitForTimeout(900)
await page2.locator('button', { hasText: /^teleop$/ }).first().click()
await page2.waitForTimeout(1000)
await page2.locator('button', { hasText: /^stop teleop$/ }).first().click()
await page2.waitForTimeout(300)
await page2.locator('button', { hasText: /^confirm — stop teleop/ }).first().click()
await page2.waitForTimeout(1800)
const stubborn = (await page2.locator('div.hint').filter({ hasText: 'teleop' }).first().innerText().catch(() => '')).replace(/\s+/g, ' ')
if (/teleop stopped/.test(stubborn) || !/STILL on the wire/.test(stubborn))
  failures.push(`a stop that did not take was reported as done: ${stubborn.slice(0, 160)}`)
else console.log(`  note  a stop that did not take says: ${stubborn.match(/stop was sent[^·]*/)?.[0]?.slice(0, 110)}`)

if (seen !== handled) failures.push(`ISOLATION BROKEN: ${seen - handled} teleop request(s) escaped to the real fleet — this audit must not touch hardware`)
else console.log(`  note  isolation proven: ${handled} teleop request(s), all intercepted, none forwarded`)

await browser.close()
if (failures.length) { for (const f of failures) console.error(`  FAIL  ${f}`); process.exit(1) }
console.log('  ok    a stream is visible, stopping is two-step, and only the arm may call it stopped')
