/**
 * The refusals a record session can be continued past — and the ones it cannot.
 *
 * Every camera gate at /api/record/open is continuable by design; this pins that
 * the continuation is REACHABLE (the flag the server names becomes a tick) and
 * that the allowlist keeps a bypass from appearing where nobody designed one.
 */
import assert from 'node:assert/strict'
import { overrideOffered, nextAcknowledged, overrideBodyFlags } from '/tmp/recordRefusal.mjs'

const ack = (offered, ticked, prev = []) => overrideBodyFlags(nextAcknowledged(prev, offered, ticked))

// Real server sentences, not paraphrases: this module reads what the backend writes.
const STALE =
  "so101-arm-1: 1 configured camera stopped publishing - wrist (last frame 10.4h ago). " +
  "Pass ignore_dead_cameras to record anyway."
const GONE =
  "so101-arm-1: 1 configured camera is not listed by this machine at all - wrist (index 1). " +
  "Pass ignore_missing_cameras to record without it anyway."
const DRIFT =
  "so101-arm-1: 1 configured camera changed hands - wrist index 0 was USB2.0_CAM1, now Logi 4K Pro " +
  "(USB2.0_CAM1 is at index 1 now). Point the camera at its new index, or pass " +
  "ignore_camera_identity to record with the index as it stands."

for (const [msg, flag] of [
  [STALE, 'ignore_dead_cameras'],
  [GONE, 'ignore_missing_cameras'],
  [DRIFT, 'ignore_camera_identity'],
]) {
  const o = overrideOffered(msg)
  assert.equal(o?.flag, flag, `the refusal naming ${flag} must offer it`)
  assert.ok(o.label.length > 10 && o.cost.length > 10, 'a tick states the claim AND the cost')
}

// The identity override is the one that permits something VISIBLY working, so its
// cost line must say the consequence out loud rather than reassure.
assert.match(overrideOffered(DRIFT).cost, /WRONG view/)

// No flag named -> no tick. A bypass must never appear on a message nobody designed.
for (const msg of [
  'a recording session is already open - close it first',
  'no teleoperator type known for leader robot',
  'Failed to fetch',
  '',
  null,
  undefined,
  42,
  { detail: 'ignore_dead_cameras' },
]) {
  assert.equal(overrideOffered(msg), null)
}

// A word that merely looks like one is not one.
assert.equal(overrideOffered('pass ignore_dead_camera to proceed'), null)
assert.equal(overrideOffered('ignoring dead cameras is not an option'), null)

// TWO faults refused at once are two admissions; one box would collect consent
// for the one the operator did not read.
assert.equal(overrideOffered(`${STALE} ${DRIFT}`), null)

// The flag travels only with a deliberate tick, and never by default.
const drift = overrideOffered(DRIFT)
assert.deepEqual(ack(drift, true), { ignore_camera_identity: true })
assert.deepEqual(ack(drift, false), {}, 'an unticked box sends nothing')
assert.deepEqual(ack(null, true), {}, 'a tick with nothing offered sends nothing')

console.log('recordRefusal: ok')

// ── Q98: the admissions ACCUMULATE, or a two-fault camera loops forever ──
// The route checks dead -> missing -> identity, and each gate is skipped only by its own flag. Sending
// just the last refusal's flag ping-ponged: ignore_missing gets past gate 2 and is then refused by
// gate 3, whose retry drops ignore_missing and is refused by gate 2 again. And that pair is ONE
// PHYSICAL EVENT - unplugging a camera makes it missing and renumbers the rest into identity drift -
// so the commonest two-fault case could not be continued from the screen at all.
const missing = overrideOffered(GONE)
const identity = overrideOffered(DRIFT)
assert.equal(missing.flag, 'ignore_missing_cameras')
assert.equal(identity.flag, 'ignore_camera_identity')

const afterFirst = nextAcknowledged([], missing, true)
assert.deepEqual(afterFirst, ['ignore_missing_cameras'])
const afterSecond = nextAcknowledged(afterFirst, identity, true)
assert.deepEqual(afterSecond, ['ignore_missing_cameras', 'ignore_camera_identity'],
  'THE REGRESSION: the earlier admission must survive the next refusal, or the gates ping-pong')
assert.deepEqual(overrideBodyFlags(afterSecond),
  { ignore_missing_cameras: true, ignore_camera_identity: true }, 'and both are actually sent')

// Order is the ROUTE's (dead, missing, identity), not the order the faults happened to be met in - in
// the SET as well as the body. Asserting it on the body alone proved nothing: overrideBodyFlags walks
// the allowlist itself, so a body is canonical however the set was built. (A surviving mutation said
// so - the assertion, not the code, was the weak half.)
assert.deepEqual(nextAcknowledged(['ignore_camera_identity'], missing, true),
                 ['ignore_missing_cameras', 'ignore_camera_identity'])
assert.deepEqual(Object.keys(overrideBodyFlags(['ignore_camera_identity', 'ignore_missing_cameras'])),
                 ['ignore_missing_cameras', 'ignore_camera_identity'])

// An UNTICKED refusal adds nothing, and never removes what was already admitted: not answering the
// new question is not withdrawal of the old answer.
assert.deepEqual(nextAcknowledged(afterFirst, identity, false), afterFirst)
// Ticking the same refusal twice cannot grow a duplicate.
assert.deepEqual(nextAcknowledged(afterFirst, missing, true), afterFirst)

// THE ALLOWLIST STILL RULES: the set is not a place a flag can arrive from anywhere else. Anything
// that is not one of the three - a typo, a server-invented field, an injected key - is dropped.
assert.deepEqual(nextAcknowledged(['ignore_everything', 'ignore_dead_camera', '__proto__'], null, false), [])
assert.deepEqual(overrideBodyFlags(['ignore_everything', 'constructor']), {})
assert.deepEqual(overrideBodyFlags([]), {}, 'no admissions is an empty body, not a missing one')

// A message naming TWO flags still offers NO tick: that rule is about one refusal being two
// admissions, and accumulating over SEPARATE refusals does not weaken it.
assert.equal(overrideOffered(`${GONE} ${DRIFT}`), null)

console.log('recordRefusal: Q98 accumulation assertions ok')
