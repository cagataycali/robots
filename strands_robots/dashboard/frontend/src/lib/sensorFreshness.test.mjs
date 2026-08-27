import assert from 'node:assert/strict'
import {
  sensorVerdict, declaredKinds, rowsToShow, stripSummary, SENSOR_QUIET_S, SENSOR_KINDS,
} from '/tmp/sensorFreshness.mjs'

const NOW = 1787195402.7

// --- rail 1: the peer's own declaration --------------------------------------
// `presence.topics` is built by mesh.core from the providers it found on the robot.
{
  assert.deepEqual(declaredKinds(['pose', 'imu', 'odom', 'lidar', 'health']),
    ['health', 'pose', 'odom', 'imu', 'lidar'], 'returned in reading order, health first')
  assert.deepEqual(declaredKinds(['health']), ['health'], 'a joints-only arm declares exactly this')
  // 'hand' and 'map' are declared by the SDK and not carried by this strip: they must not appear.
  assert.deepEqual(declaredKinds(['hand', 'map']), [])
  // An older peer sends no `topics` at all. Nothing may be inferred from a field it never sent.
  assert.deepEqual(declaredKinds(undefined), [])
  assert.deepEqual(declaredKinds(null), [])
  assert.deepEqual(declaredKinds('pose'), [], 'a non-array is not a declaration')
}

// --- the three-way distinction the rails exist for ---------------------------
{
  // never declared, never seen: not this robot's business
  const absent = sensorVerdict('lidar', null, NOW, false)
  assert.equal(absent.tone, 'absent')
  assert.equal(absent.text, 'not published by this robot')
  assert.equal(absent.ageS, null)

  // DECLARED and silent: a real finding, and NOT the same sentence as above
  const waiting = sensorVerdict('lidar', null, NOW, true)
  assert.equal(waiting.tone, 'waiting')
  assert.match(waiting.text, /declared, waiting for the first reading/)
  assert.doesNotMatch(waiting.text, /not published/,
    'the false equivalence this module exists to prevent')

  // arriving
  const live = sensorVerdict('lidar', { t: NOW - 0.2 }, NOW, true)
  assert.equal(live.tone, 'live')

  // was arriving, stopped
  const stale = sensorVerdict('lidar', { t: NOW - 600 }, NOW, true)
  assert.equal(stale.tone, 'stale')
  assert.match(stale.text, /last reading 10m ago/)
}

// --- `declared` defaults to the neutral reading ------------------------------
assert.equal(sensorVerdict('pose', null, NOW).tone, 'absent',
  'omitting the rail must not manufacture a complaint')

// --- freshness window --------------------------------------------------------
{
  assert.equal(sensorVerdict('pose', { t: NOW - SENSOR_QUIET_S + 0.01 }, NOW).tone, 'live')
  assert.equal(sensorVerdict('pose', { t: NOW - SENSOR_QUIET_S - 0.01 }, NOW).tone, 'stale')
  assert.match(sensorVerdict('imu', { t: NOW - 7200 }, NOW).text, /2\.0h ago/)
  assert.match(sensorVerdict('imu', { t: NOW - 45 }, NOW).text, /45s ago/)
}

// --- a payload that arrived without a timestamp is NOT "not published" -------
for (const bad of [{}, { t: null }, { t: 0 }, { t: 'soon' }, { t: NaN }, { t: Infinity }]) {
  const v = sensorVerdict('health', bad, NOW, true)
  assert.equal(v.tone, 'live', `t=${JSON.stringify(bad.t)} still arrived`)
  assert.doesNotMatch(v.text, /not published|waiting/)
  assert.equal(v.ageS, null, 'no timestamp means no honest age')
}

// --- clock skew must not invent a quiet period -------------------------------
{
  const v = sensorVerdict('pose', { t: NOW + 5 }, NOW, true)
  assert.equal(v.tone, 'live')
  assert.equal(v.ageS, 0, 'a peer ahead of us is current, not negative-aged')
}

// --- which rows to draw ------------------------------------------------------
{
  // declared but silent still gets a row: that IS the finding.
  assert.deepEqual(rowsToShow(['pose', 'lidar'], {}), ['pose', 'lidar'])
  // arrived without being declared (older peer) also gets one.
  assert.deepEqual(rowsToShow(null, { imu: { t: 1 } }), ['imu'])
  assert.deepEqual(rowsToShow(['health'], { health: { t: 1 } }), ['health'], 'no duplicates')
  assert.deepEqual(rowsToShow(null, null), [])
  assert.deepEqual(rowsToShow(['pose', 'health'], { imu: { t: 1 } }),
    ['health', 'pose', 'imu'], 'union, in reading order')
}

// --- an arm: measured `topics: ['health']` and a real host-stats payload -----
{
  const arm = { health: { t: NOW - 0.4, cpu_load: 2.71, disk_free_gb: 86.7 } }
  const rows = rowsToShow(['health'], arm)
  assert.deepEqual(rows, ['health'], 'one row, not five')
  const v = rows.map(k => sensorVerdict(k, arm[k], NOW, true))
  assert.equal(stripSummary(v).text, '1 sensor arriving')
}

// --- a peer with nothing at all renders NOTHING ------------------------------
{
  const none = SENSOR_KINDS.map(k => sensorVerdict(k, null, NOW, false))
  assert.equal(stripSummary(none), null, 'a permanent "no lidar" line on every card is noise')
}

// --- the summary names what is wrong, and never accuses the absent -----------
{
  const rover = [
    sensorVerdict('health', { t: NOW }, NOW, true),
    sensorVerdict('pose', { t: NOW - 300 }, NOW, true),
    sensorVerdict('imu', null, NOW, false),
  ]
  const s = stripSummary(rover)
  assert.equal(s.tone, 'stale')
  assert.equal(s.text, 'pose went quiet')
  assert.doesNotMatch(s.text, /imu/, 'the undeclared imu is not part of the complaint')
}
{
  // declared, none arrived yet: neutral, and it says so rather than claiming a fault
  const boot = [sensorVerdict('pose', null, NOW, true), sensorVerdict('lidar', null, NOW, true)]
  const s = stripSummary(boot)
  assert.equal(s.tone, 'waiting')
  assert.match(s.text, /pose, lidar declared, nothing yet/)
}
{
  // some live, some still waiting
  const mixed = [sensorVerdict('health', { t: NOW }, NOW, true), sensorVerdict('lidar', null, NOW, true)]
  const s = stripSummary(mixed)
  assert.equal(s.tone, 'live')
  assert.equal(s.text, '1 sensor arriving, 1 not started')
}

console.log('ok sensorFreshness')
