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
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
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
for (let i = 0; i < ports.length; i++) {
  const closed = page.locator('button:has-text("calibrate…")')
  if (!(await closed.count())) break
  await closed.first().click()
  await page.waitForTimeout(1200)
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
  if (!text) { failures.push(`${label}: no calibrate panel rendered for this row`); continue }
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

await browser.close()
if (failures.length) { console.log('\nFAILURES:'); for (const f of failures) console.log(`  ✗ ${f}`); process.exit(1) }
console.log('\ncalibrate command: the page names the id every arm actually loads')
