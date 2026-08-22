import assert from 'node:assert/strict'

const { rememberedLine, nameClaimsOtherRole } = await import('/tmp/rememberedBoard.mjs')

// --- nothing remembered renders as nothing: an unconfigured board is a normal state ---
assert.equal(rememberedLine(null, {}), null)
assert.equal(rememberedLine({ peer_id: '', cameras: [] }, {}), null)

// --- the ordinary line ---
{
  const l = rememberedLine(
    { peer_id: 'so101-arm-1', robot_name: 'so101', mode: 'real', cameras: ['top', 'wrist'], robot_id: 'arm_1' },
    { role: 'leader', role_volts: 7.4 },
  )
  assert.equal(l.summary, 'so101-arm-1 — so101, real, cameras top + wrist')
  assert.equal(l.calibrationId, 'arm_1')
  assert.equal(l.warning, undefined, 'a neutral id must not raise a warning')
  // Camera indices must never appear: macOS renumbers them, so they would be confidently stale.
  assert.ok(!/index/.test(JSON.stringify(l)))
}

// --- THE POINT OF THE FILE: the real profile on cagatay's desk ---
{
  // arm-2 measures 12.6V = FOLLOWER, and its saved profile carries robot_id "leader_arm".
  const l = rememberedLine(
    { peer_id: 'so101-arm-2', robot_name: 'so101', mode: 'real', cameras: [], robot_id: 'leader_arm' },
    { role: 'follower', role_volts: 12.6 },
  )
  assert.ok(l.warning, 'a leader-named id on a measured follower must be called out')
  assert.match(l.warning, /12\.6V = follower/)
  assert.match(l.warning, /name is what is wrong/)
  // It is a NOTE, not a refusal: the calibration file really does live under that id.
  assert.match(l.warning, /reuse the memory anyway/)
  assert.equal(l.calibrationId, 'leader_arm', 'the id is still shown - it is what lerobot will read')
}

// --- a peer NAME can carry the same lie ---
{
  const l = rememberedLine({ peer_id: 'leader-arm', cameras: [] }, { role: 'follower', role_volts: 12.7 })
  assert.match(l.warning, /this peer is named "leader-arm"/)
}

// --- an INDEX in a name is never evidence (armPairing's rule, same reasoning) ---
assert.equal(nameClaimsOtherRole('so101-arm-2', 'follower'), false)
assert.equal(nameClaimsOtherRole('arm_1', 'leader'), false)
// A name containing BOTH words is ambiguous, not a contradiction.
assert.equal(nameClaimsOtherRole('leader_follower_rig', 'follower'), false)
assert.equal(nameClaimsOtherRole('follower_arm', 'follower'), false)
assert.equal(nameClaimsOtherRole('LEADER_ARM', 'follower'), true)

// --- silence without a measurement: an unmeasured bus cannot contradict anything ---
{
  const l = rememberedLine({ peer_id: 'leader-arm', cameras: [], robot_id: 'leader_arm' }, {})
  assert.equal(l.warning, undefined, 'no measured role means no contradiction to report')
}
{
  // "unpowered" / "mixed" are verdicts, not roles - they must not be compared against a name either.
  const l = rememberedLine({ peer_id: 'leader-arm', cameras: [] }, { role: 'unpowered', role_volts: 5.5 })
  assert.equal(l.warning, undefined)
}

// --- a memory with no voltage recorded still says something true ---
{
  const l = rememberedLine({ peer_id: 'leader_arm', cameras: [] }, { role: 'follower' })
  assert.match(l.warning, /its measured voltage = follower/)
}

// --- THE SECOND TRAP: a remembered id lerobot cannot load (measured on the live rig) ---
const CALS = [
  { id: 'leader', deviceType: 'teleoperators', model: 'so101_leader', motors: 6 },
  { id: 'follower', deviceType: 'robots', model: 'so101_follower', motors: 6 },
  { id: 'leader_arm', deviceType: 'robots', model: 'so101_follower', motors: 6 },
]
{
  // so101-leader's real profile: robot_id 'leader', which exists only teleoperator-side.
  const l = rememberedLine(
    { peer_id: 'so101-leader', robot_name: 'so101', mode: 'real', robot_id: 'leader', cameras: ['main'] },
    { calibrations: CALS })
  assert.match(l.idProblem, /calibrated as a teleoperator/)
  assert.match(l.idProblem, /repeats that failure/, 'the point is that ONE CLICK reproduces it')
  assert.match(l.idProblem, /leader_arm/, 'and which ids would work instead')
  assert.equal(l.warning, undefined, 'nothing was measured here, so no name/role claim is made')
}
{
  // A loadable id says nothing at all - this row is dense already.
  const l = rememberedLine(
    { peer_id: 'so101-follower', robot_id: 'follower', cameras: [] }, { calibrations: CALS })
  assert.equal(l.idProblem, undefined)
}
{
  // An id that exists on NEITHER side is just as doomed, and gets named too.
  const l = rememberedLine({ peer_id: 'a', robot_id: 'nope', cameras: [] }, { calibrations: CALS })
  assert.match(l.idProblem, /no calibration named "nope"/)
}
{
  // The list has not arrived: claim nothing. Accusing a memory on absent evidence would scare an
  // operator off the one button that fixes their board.
  for (const cals of [undefined, null]) {
    const l = rememberedLine({ peer_id: 'a', robot_id: 'leader', cameras: [] }, { calibrations: cals })
    assert.equal(l.idProblem, undefined)
  }
}
{
  // No remembered id at all: the spawn form's "no id" warning belongs to the form, not to a row
  // describing what once happened.
  const l = rememberedLine({ peer_id: 'a', cameras: [] }, { calibrations: CALS })
  assert.equal(l.idProblem, undefined)
}
{
  const l = rememberedLine(
    { peer_id: 'so101-leader', robot_id: 'leader', cameras: [] },
    { role: 'follower', role_volts: 12.6, calibrations: CALS })
  assert.match(l.warning, /12.6V = follower/)
  assert.match(l.idProblem, /calibrated as a teleoperator/)
}

console.log('rememberedBoard: ok')
