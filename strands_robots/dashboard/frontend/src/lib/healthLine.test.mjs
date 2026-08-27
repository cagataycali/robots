import assert from 'node:assert/strict'
import {
  healthLine, hottest, uptimeText, BATTERY_LOW_PCT, DISK_CRITICAL_GB, DISK_TIGHT_GB,
} from '/tmp/healthLine.mjs'

// --- no topic is not a fault -------------------------------------------------
{
  const l = healthLine(null)
  assert.equal(l.tone, 'none')
  assert.match(l.text, /no health topic on this robot/)
  assert.equal(l.detail, null)
  assert.equal(healthLine(undefined).tone, 'none')
}

// --- the SDK can publish a health payload with a NULL battery ------------------
// `_read_health` sets has_data for a battery dict carrying neither pct nor percentage.
{
  const l = healthLine({ peer_id: 'x', t: 1, battery_pct: null, charging: false })
  assert.equal(l.tone, 'none', 'nothing readable is not an accusation')
  assert.match(l.text, /reported no readings/)
  assert.equal(l.detail, null)
}

// --- a healthy robot reports the reading asked for first ----------------------
{
  const l = healthLine({ battery_pct: 61.4, charging: false, disk_free_gb: 210.5, cpu_load: 2.14 })
  assert.equal(l.tone, 'ok')
  assert.equal(l.text, 'battery 61%')
  assert.match(l.detail, /battery 61%/)
  assert.match(l.detail, /210\.5 GB free/)
  assert.match(l.detail, /load 2\.14/)
}
{
  const l = healthLine({ battery_pct: 61, charging: true })
  assert.equal(l.text, 'battery 61%, charging')
  assert.match(l.detail, /\(charging\)/)
}

// --- a low battery only complains while DISCHARGING ---------------------------
{
  const low = healthLine({ battery_pct: BATTERY_LOW_PCT, charging: false })
  assert.equal(low.tone, 'attention')
  assert.match(low.text, /battery 10% and discharging/)
  const plugged = healthLine({ battery_pct: BATTERY_LOW_PCT, charging: true })
  assert.equal(plugged.tone, 'ok', '8% on the charger is a robot that is fine')
  const justOver = healthLine({ battery_pct: BATTERY_LOW_PCT + 1, charging: false })
  assert.equal(justOver.tone, 'ok')
}

// --- disk reuses the dashboard's OWN floors ------------------------------------
{
  const dead = healthLine({ disk_free_gb: DISK_CRITICAL_GB - 0.1 })
  assert.equal(dead.tone, 'attention')
  assert.match(dead.text, /not enough to finish a recording/)
  const tight = healthLine({ disk_free_gb: DISK_TIGHT_GB - 1 })
  assert.equal(tight.tone, 'attention')
  assert.match(tight.text, /tight for a long session/)
  const fine = healthLine({ disk_free_gb: DISK_TIGHT_GB + 1 })
  assert.equal(fine.tone, 'ok')
}

// --- battery outranks disk: a robot that is about to stop moving wins ---------
{
  const l = healthLine({ battery_pct: 4, charging: false, disk_free_gb: 0.5 })
  assert.match(l.text, /battery 4%/)
  assert.match(l.detail, /0\.5 GB free/, 'the disk fact is still reachable')
}

// --- a robot reporting only host stats still gets a line ----------------------
{
  const l = healthLine({ cpu_load: 0.5, mem_pct: 31.2, uptime_s: 12600 })
  assert.equal(l.tone, 'ok')
  assert.match(l.text, /memory 31%/, 'the first reported reading leads')
  assert.match(l.detail, /up 3\.5h/)
}

// --- non-finite numbers are not readings -------------------------------------
for (const bad of [NaN, Infinity, -Infinity, '61', null, undefined]) {
  const l = healthLine({ battery_pct: bad })
  assert.equal(l.tone, 'none', `battery_pct=${String(bad)} is not a reading`)
}

// --- helpers -----------------------------------------------------------------
assert.equal(hottest(null), null)
assert.equal(hottest({}), null)
assert.deepEqual(hottest({ cpu: 41, motor: 63.5, board: 39 }), ['motor', 63.5])
assert.deepEqual(hottest({ cpu: 41, bad: 'hot' }), ['cpu', 41], 'a junk entry is skipped, not fatal')
assert.equal(uptimeText(45), '45s')
assert.equal(uptimeText(600), '10m')
assert.equal(uptimeText(7200), '2.0h')
assert.equal(uptimeText(-1), null)
assert.equal(uptimeText(null), null)

console.log('ok healthLine')
