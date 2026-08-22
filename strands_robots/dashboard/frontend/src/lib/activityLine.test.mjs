import assert from 'node:assert/strict'
import { activityLine } from '/tmp/activityLine.mjs'

const q30 = activityLine({
  action: 'estop', target: '', ok: true,
  detail: { peer_id: 'evac-coordinator', responses_received: 0, peers_not_stopped: [], lockout_engaged: true },
})
assert.notEqual(q30.tone, 'ok', 'an unacknowledged emergency stop is not a success row')
assert.equal(q30.tone, 'warn')
assert.equal(q30.glyph, '⚠')
assert.match(q30.title, /broadcast but no peer confirmed/)
// WHO fired it is the first question anyone asks in an incident.
assert.match(q30.note, /issued by evac-coordinator/)
assert.match(q30.note, /lockout engaged/)
assert.match(q30.note, /no peer acknowledged/)
// An empty target read as a missing value; a fleet-wide stop has a name.
assert.equal(q30.target, 'all peers')

// An e-stop everything acknowledged is still not a green ✓ - it is an event that
// stopped a fleet, and the log should read like one.
const clean = activityLine({
  action: 'estop', target: '', ok: true,
  detail: { peer_id: 'dashboard', responses_received: 2, peers_not_stopped: [], lockout_engaged: true },
})
assert.equal(clean.tone, 'bad')
assert.equal(clean.glyph, '■')
assert.match(clean.note, /2 peers acknowledged/)
assert.match(clean.title, /the fleet was stopped/)
assert.equal(activityLine({ action: 'estop', ok: true, detail: { responses_received: 1 } }).note,
  '1 peer acknowledged', 'singular')

// A peer that did NOT stop is the worst case and must be counted on the line.
const partial = activityLine({
  action: 'estop', target: '', ok: true,
  detail: { responses_received: 3, peers_not_stopped: ['so101-arm-2'], lockout_engaged: true },
})
assert.equal(partial.tone, 'warn')
assert.match(partial.note, /1 did NOT stop/)

// --- "the call completed" is not "the robot answered" (Q7 family) ------------
for (const row of [
  { action: 'task', target: 'so101-arm-1', ok: true, detail: { state: 'no_answer' } },
  { action: 'stop', target: 'so101-arm-1', ok: true, result: '{"state": "no_answer"}' },
  { action: 'teleop', target: 'so101-arm-1', ok: true, detail: { answered: false } },
]) {
  const v = activityLine(row)
  assert.equal(v.tone, 'warn', `no_answer must not be a green tick: ${JSON.stringify(row)}`)
  assert.equal(v.glyph, '⚠')
  assert.equal(v.note, 'robot did not answer')
  assert.match(v.title, /never answered/)
}

// --- everything else is unchanged -------------------------------------------
const good = activityLine({ action: 'spawn', target: 'sim-a', ok: true, detail: { pid: 42 } })
assert.equal(good.tone, 'ok'); assert.equal(good.glyph, '✓'); assert.equal(good.note, '')
assert.equal(good.target, 'sim-a')
const failed = activityLine({ action: 'spawn', target: 'sim-a', ok: false, detail: { error: 'boom' } })
assert.equal(failed.tone, 'bad'); assert.equal(failed.glyph, '✗')
const pending = activityLine({ action: 'task', target: 'sim-a', ok: null })
assert.equal(pending.tone, 'pending'); assert.equal(pending.glyph, '…')
// A failed call outranks the e-stop wording: ✗ is the truth there.
assert.equal(activityLine({ action: 'estop', ok: false, detail: {} }).glyph, '✗')
// A non-e-stop row with no target says so instead of rendering an empty box.
assert.equal(activityLine({ action: 'rescan', target: '', ok: true }).target, '—')

// Malformed detail must never crash a log row (it is a live incident surface).
for (const d of [null, undefined, 'a string', 42, ['list'], { responses_received: 'two' }]) {
  const v = activityLine({ action: 'estop', ok: true, detail: d })
  assert.ok(['warn', 'bad'].includes(v.tone))
  assert.ok(typeof v.note === 'string')
}
assert.ok(activityLine({ action: '', ok: true }).glyph)

console.log('activityLine: 40 assertions ok')
