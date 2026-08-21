/**
 * The calibrate command ON THE PAGE names the id the arm actually loads.
 *
 * R5 hands the operator a prefilled `lerobot-calibrate` line. The id in it is the one thing that
 * cannot be checked by reading the code: `deviceId()` honours a spawn profile's `robot_id` when
 * there is one, but that value has to travel scan row -> profiles fetch -> plan -> DOM, and an
 * earlier version of this feature was CORRECT in the lib and inert in the UI (572bb706 fixed the
 * lib, dc303848 discovered nothing passed it). So this audit compares two independent sources: the
 * command text the browser renders, and the profile store read straight from the API.
 *
 * Why the id matters more than the other three flags: lerobot loads calibration BY id. A command
 * naming a different id sends the operator through the whole ceremony - moving every joint to its
 * limits by hand - and writes a file the running arm never opens. Nothing reports a failure; the
 * only symptom is a real arm still reaching where it should not.
 *
 * READ-ONLY: it opens the devices screen and toggles a disclosure. It never spawns, records,
 * measures or moves anything, so it is safe against the live rig with the arms powered.
 *
 * Run: node scripts/audit-calibrate-command.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'
import { apiSettled, SCREEN_APIS } from './_audit_wait.mjs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const api = async (path) => {
  const r = await fetch(`${BASE}${path}`, { headers: { Authorization: `Bearer ${TOKEN}` } })
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`)
  return r.json()
}

const [devices, profileDoc] = await Promise.all([api('/api/devices'), api('/api/devices/profiles')])
const profiles = profileDoc?.profiles ?? {}
/** The id the profile store remembers for a port - the same rule as lib/calibrateCommand.ts. */
const remembered = (port, serial) => {
  const bySerial = (profiles[serial ?? '']?.robot_id ?? '').trim()
  if (bySerial) return bySerial
  for (const p of Object.values(profiles)) if ((p?.port ?? '').trim() === port) {
    const id = (p?.robot_id ?? '').trim()
    if (id) return id
  }
  return null
}

const browser = await chromium.launch()
const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage()
page.on('pageerror', e => failures.push(`page threw: ${String(e.message).slice(0, 160)}`))
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(6000)
/* The devices screen's calibration verdicts come from /api/calibration and /api/devices, both
   fetched after the panel mounts: read the DOM before they land and the audit reports the
   product's words wrong. See _audit_wait.mjs for the entry that cost a retraction. */
const devicesReady = apiSettled(page, ...SCREEN_APIS.devices)
await page.locator('button.chip:has-text("devices")').first().click()
await devicesReady
await page.waitForTimeout(1500)

const ports = devices.serial_ports ?? []
console.log(`${ports.length} serial port(s) on this rig`)

// One row at a time, keyed by the row's OWN port. Both facts here were measured, not assumed:
// the panel is a single-value disclosure (`calibFor` holds one port, so opening the second row
// closes the first), and reading `querySelector('.calibcmd')` globally attributed one arm's
// command to the other in the first version of this script. Keying by the row makes a mis-scoped
// read impossible rather than unlikely - the same reason the lib refuses a prefix port match.
const panels = {}
/* PRECONDITION, stated and measured. Under a full `audit:all` sweep this script reported "no
   calibrate panel rendered for this row" for BOTH ports while the product was fine: the devices
   screen had not painted its rows yet (30 audits back-to-back keep the server busy), the loop below
   found zero `calibrate…` buttons, silently `break`ed, and the verdict loop blamed the page for
   panels it had never even tried to open. A precondition that fails silently turns into a false
   accusation against the thing being audited, so it is checked FIRST and reported in its own words. */
const wantRows = ports.length
let paintedRows = await page.locator('button:has-text("calibrate…")').count()
if (paintedRows < wantRows) {
  await page.waitForFunction(
    n => document.querySelectorAll('li').length && [...document.querySelectorAll('button')]
      .filter(b => b.innerText.includes('calibrate…') || b.innerText.includes('hide calibrate command')).length >= n,
    wantRows, { timeout: 20000 }).catch(() => {})
  paintedRows = await page.locator('button:has-text("calibrate…")').count()
}
if (paintedRows < wantRows) {
  failures.push(`the devices screen painted ${paintedRows} calibrate control(s) for the ${wantRows} serial `
    + 'port(s) /api/devices reports, after its APIs settled and 20s of waiting — the page\'s calibrate '
    + 'command was never examined, so this is a PRECONDITION failure (a busy server, most likely under a '
    + 'full sweep), NOT a missing panel')
}

const timedOut = new Set()
for (let i = 0; i < ports.length; i++) {
  const closed = page.locator('button:has-text("calibrate…")')
  if (!(await closed.count())) break
  await closed.first().click()
  // WAIT for the panel, do not guess at it: this script failed once in five runs with "no calibrate
  // panel rendered for this row" while the product was fine. The disclosure is single-value and its
  // content arrives with a fetch, so a flat sleep is a race whose loser blames the page. A flake
  // here is expensive twice over — it reads as a product defect AND it teaches people to rerun.
  /* The panel renders synchronously from props (DevicePanel reads `calibFor === p.device`), so a
     timeout here means the CLICK did not take, never that content is slow — and it must say so
     instead of being swallowed into an empty panel map. */
  await page.locator('.calibcmd').first().waitFor({ state: 'visible', timeout: 15000 })
    .catch(() => { timedOut.add(i) })
  await page.waitForTimeout(300)
  Object.assign(panels, await page.evaluate(() => {
    const out = {}
    for (const li of document.querySelectorAll('li')) {
      const cmd = li.querySelector('.calibcmd')
      const port = li.querySelector('.mono')?.innerText?.trim()
      if (cmd && port) out[port] = cmd.innerText.replace(/\n+/g, ' | ')
    }
    return out
  }))
}

for (const p of ports) {
  const text = panels[p.device] ?? ''
  const role = (p.role ?? '').trim()
  const label = `${p.device} (${role || 'unmeasured'})`
  if (!text) {
    failures.push(timedOut.size
      ? `${label}: the calibrate disclosure did not become visible within 15s of the click — a TIMEOUT `
        + '(the panel renders from props, so the click did not take), not a page that renders no command'
      : `${label}: the row opened but rendered no calibrate command`)
    continue
  }
  const cmd = /lerobot-calibrate[^|]*/.exec(text)?.[0]?.trim() ?? null
  const want = remembered(p.device, p.serial_number)

  if (!role) {
    // An unmeasured bus must NOT produce a command: the role decides the model AND the
    // directory, so a guess writes a real file in the wrong place.
    if (cmd) failures.push(`${label}: rendered a command with no measured role -> ${cmd}`)
    else if (!/measure/.test(text)) failures.push(`${label}: refused without telling the operator to measure`)
    else console.log(`  ok    ${label}: no command, and it says to measure first`)
    continue
  }
  if (!cmd) { console.log(`  note  ${label}: no command (${text.slice(0, 90)})`); continue }

  const got = /--device_id=('?)([^ '|]+)\1/.exec(cmd)?.[2] ?? null
  if (!/--port=/.test(cmd) || !cmd.includes(p.device)) {
    failures.push(`${label}: the command does not name this row's port -> ${cmd}`)
  }
  if (want && got !== want) {
    failures.push(`${label}: command says --device_id=${got} but this arm LOADS ${want}`)
  } else if (want) {
    console.log(`  ok    ${label}: --device_id=${got}, the id the arm loads`)
  } else if (!/no spawn profile/.test(text)) {
    failures.push(`${label}: invented the id ${got} without saying the port has no profile`)
  } else {
    console.log(`  ok    ${label}: invented ${got} and said why`)
  }
  // A correct id whose NAME says the other role must be flagged, or a human reads the
  // name instead of the flags and believes they are calibrating the other arm.
  const other = role === 'follower' ? 'leader' : 'follower'
  if (got && got.toLowerCase().includes(other) && !text.includes('\u26a0')) {
    failures.push(`${label}: id ${got} contradicts the measured role and nothing warns`)
  }
}

// ---- Q131: the same disclosure at phone width. A calibrate command is the one thing on this
// screen an operator must READ CHARACTER BY CHARACTER (it names a /dev path and a robot id), and it
// is long, monospaced and machine-generated — the exact shape that overflows a 390px column. The
// CSS says it was designed for this (overflow-wrap: anywhere, user-select: all, because clipboard
// access is unavailable on a non-secure LAN origin and selecting the text is then the ONLY route),
// so this proves the design holds rather than assuming it.
{
  const PHONE = { width: 390, height: 844 }
  const ctx = await browser.newContext({ viewport: PHONE, serviceWorkers: 'block' })
  const ph = await ctx.newPage()
  ph.on('pageerror', e => failures.push(`phone page threw: ${String(e.message).slice(0, 160)}`))
  await ph.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await ph.waitForTimeout(6000)
  const ready = apiSettled(ph, ...SCREEN_APIS.devices)
  await ph.locator('button.chip:has-text("devices")').first().click()
  await ready
  await ph.waitForTimeout(1500)
  const opener = ph.locator('button:has-text("calibrate…")')
  if (!(await opener.count())) {
    // Not a failure: with no serial port on the rig there is no disclosure to measure, and an
    // audit that fails for absent hardware is noise that trains people to ignore it.
    console.log('  SKIP  phone pass — no calibrate row on this rig')  // SKIP-prefixed so the sweep counts it as narrowing (2026-08-22): lowercase prose 'skipped' was invisible to the runner's summary.
  } else {
    await opener.first().click()
    await ph.waitForTimeout(1200)
    const m = await ph.evaluate(() => {
      const d = document.documentElement
      const code = document.querySelector('.calibcmd .cmdline')
      const copy = [...document.querySelectorAll('.calibcmd .row button')]
        .find(b => /copy/i.test(b.innerText))
      const box = code?.getBoundingClientRect()
      return {
        pageWide: d.scrollWidth > d.clientWidth + 1, scroll: d.scrollWidth, client: d.clientWidth,
        hasCode: !!code,
        text: code?.innerText ?? '',
        wraps: code ? code.scrollWidth <= code.clientWidth + 1 : false,
        right: box ? box.x + box.width : 0,
        lines: box && code ? Math.round(box.height / parseFloat(getComputedStyle(code).lineHeight)) : 0,
        copyVisible: !!copy && copy.getBoundingClientRect().width > 0,
      }
    })
    if (m.pageWide) {
      failures.push(`phone: the devices screen scrolls SIDEWAYS at ${PHONE.width}px `
        + `(${m.scroll} > ${m.client}) — the command runs off the edge instead of wrapping`)
    }
    if (!m.hasCode) failures.push('phone: the calibrate command line did not render')
    else {
      if (!m.wraps) {
        failures.push('phone: the command line does NOT wrap — a /dev path read through a '
          + 'horizontal scroll inside a code block is how the wrong port gets typed')
      }
      if (m.right > PHONE.width + 1) failures.push(`phone: the command extends past the viewport (right edge ${Math.round(m.right)})`)
      if (!/lerobot-calibrate/.test(m.text)) failures.push(`phone: the command lost its verb: ${JSON.stringify(m.text.slice(0, 70))}`)
      // Checked as a SHAPE, not as remembered flag names: the first version of this looked for
      // --robot.port (lerobot's own spelling) while the dashboard emits --port=, and reported a
      // missing flag on a command that was complete. `--port=` followed by a path proves the tail
      // of a five-line wrap survived, which is the property that matters.
      if (!/--port=\/\S+/.test(m.text)) {
        failures.push(`phone: no port flag with a path survived the wrap — a wrapped line must be `
          + `COMPLETE, not elided: ${JSON.stringify(m.text.slice(-60))}`)
      }
      if (!m.copyVisible) failures.push('phone: the copy button is not visible beside the command')
      console.log(`phone pass: command wraps to ~${m.lines} line(s) at ${PHONE.width}px, copy button visible`)
    }
  }
  await ctx.close()
}

await browser.close()
if (failures.length) { console.log('\nFAILURES:'); for (const f of failures) console.log(`  ✗ ${f}`); process.exit(1) }
console.log('\ncalibrate command: the page names the id every arm actually loads, and the command wraps COMPLETE '
  + 'inside a 390px column with its copy button in reach')
