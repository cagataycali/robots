/**
 * The devices screen's THREE empty worlds, proven on the page: not scanned yet, scan failed,
 * scan answered with nothing. They used to render identically ("no servo board detected",
 * "Managed robots (0) — None.", "No cameras probed — plug one in and rescan"), which told an
 * operator with two arms plugged in that his hardware was gone whenever a request failed.
 *
 * /api/devices is plain HTTP here, so page.route is the correct rail (the fleet view needs the
 * websocket instead — see audit-camera-dropped.mjs). The live dashboard is untouched: only this
 * browser's view of the endpoint is replaced. Reads only; no robot is commanded, nothing spawned.
 *
 * GOTCHA, learned while writing this: innerText returns the CSS-TRANSFORMED text, and this
 * dashboard uppercases section headings — so a case-sensitive assertion on `Managed robots (0)`
 * fails against the rendered `MANAGED ROBOTS (0)` and reads as a missing heading. Match headings
 * case-insensitively. (First version also read `.sheet, .drawer, body`, which matched a run form
 * behind the drawer; the devices screen is `aside.drawer.wide`.)
 *
 * Run: node scripts/audit-devices-empty.mjs
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`,
  'utf8',
).trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'

const EMPTY_DOC = {
  serial_ports: [], cameras: [], camera_names: [], managed: {}, camera_problem: null,
}

const fails = []
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) fails.push(label)
}

const browser = await chromium.launch()

/** Open the devices screen with /api/devices behaving as `mode`, return its text. */
async function devicesText(mode) {
  // PWA: a service-worker response is not interceptable, so the fixture would silently never land.
  const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 1000 } })
  const page = await ctx.newPage()
  await page.route(/\/api\/devices(\?|$)/, async route => {
    if (mode === 'failed') return route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"boom"}' })
    if (mode === 'pending') return new Promise(() => {})   // never answers: the first-paint world
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(EMPTY_DOC) })
  })
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(4000)
  await page.locator('button.chip:has-text("devices")').first().click()
  await page.waitForTimeout(3000)
  // The devices screen is `aside.drawer.wide`. Read THAT, not the first sheet on the page:
  // the fleet behind it has its own copy, and a body-wide read once matched a run form and
  // reported the devices heading missing.
  const panel = page.locator('aside.drawer.wide')
  await panel.first().waitFor({ timeout: 10000 })
  const text = await panel.first().innerText()
  // The `Detected hardware` inventory, read as its OWN rows: a whole-panel /^none/ also matches
  // the spawn form's `<option>none</option>`, which is a form default and not a claim about hardware.
  const rows = (await panel.first().locator('dl.kv dd').allInnerTexts()).join(' | ')
  await page.screenshot({ path: `/tmp/devices_empty_${mode}.png`, fullPage: true })
  await ctx.close()
  return { text, rows }
}

// --- world 1: the scan has not answered yet ------------------------------------
{
  const { text: t, rows } = await devicesText('pending')
  check(/scanning this machine/i.test(t), 'pending: says it is scanning', 'not a hardware verdict')
  check(!/no servo board detected/i.test(t), 'pending: does NOT claim the boards are absent')
  check(!/No camera index answered a probe/i.test(t), 'pending: does NOT claim the cameras are absent')
  check(!/plug one in and rescan/i.test(t), 'pending: does not send anyone to the cable')
  check(!/managed robots \(0\)/i.test(t), 'pending: no (0) count before an answer')
  // The `Detected hardware` inventory at the foot of the drawer — the row that gets screenshotted.
  check(/unknown — still scanning/.test(rows), 'pending: the inventory says unknown, not none', rows)
  check(!/none/i.test(rows), 'pending: no inventory row claims absent hardware', rows)
}

// --- world 2: the scan failed --------------------------------------------------
{
  const { text: t, rows } = await devicesText('failed')
  check(/this list is empty because nothing answered/i.test(t), 'failed: names the cause, not the hardware')
  check(/not because nothing is plugged in/i.test(t), 'failed: refuses the hardware claim explicitly')
  check(!/no servo board detected/i.test(t), 'failed: the old sentence is gone')
  check(!/plug one in and rescan/i.test(t), 'failed: no instruction to touch hardware')
  // The message is whatever the api helper produced for a 500 — assert the SHAPE, not its wording.
  check(/unknown — the scan failed \(.+\)/.test(rows), 'failed: the inventory names the failure', rows)
  check(!/none/i.test(rows), 'failed: no inventory row says none')
  check(!/none probed/i.test(t), 'failed: the old summary wording is gone')
  const same = t.match(/this list is empty because nothing answered/gi) ?? []
  check(same.length >= 2, 'failed: every empty list uses the SAME wording', `${same.length} occurrences`)
}

// --- world 3: the scan answered, and there is genuinely nothing ----------------
{
  const { text: t, rows } = await devicesText('empty')
  check(/no servo board detected/i.test(t), 'answered: the hardware verdict is allowed here')
  check(/nothing on USB enumerated as a serial bus/i.test(t), 'answered: says what was looked for')
  check(/No camera index answered a probe/i.test(t), 'answered: reports the probe, not the cable')
  check(/managed robots \(0\)/i.test(t), 'answered: the count is printed once it is true')
  check(!/scanning this machine/i.test(t), 'answered: no stale "scanning" line left behind')
  check(!/nothing answered, not because/i.test(t), 'answered: no failure wording on a good scan')
  check(/none \(a servo bus shows up as/.test(rows), 'answered: inventory may say none, and how it would appear')
  check(/none answered a probe/.test(rows), 'answered: cameras row reports the probe')
  check(!/unknown/i.test(rows), 'answered: nothing is left unknown once the scan spoke', rows)
}

await browser.close()
console.log(fails.length
  ? `  FAIL  ${fails.length} check(s): ${fails.join(' | ')}`
  : '  PASS  the three empty worlds are distinguishable on the page')
process.exit(fails.length ? 1 : 0)
