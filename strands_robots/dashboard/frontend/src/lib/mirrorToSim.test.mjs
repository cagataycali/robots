import assert from 'node:assert/strict'
import { mirrorPlan, mirrorSentence, twinFollowerOf } from '/tmp/mirrorToSim.mjs'

// This fleet as measured 2026-08-22: real arms WITH joints, a twin process
// with none, and the twin's robot child carrying the articulated state.
const FLEET = [
  { peer_id: 'so101-real-689', joints: 6, robot_type: 'robot', role: 'follower', role_volts: 12.6, role_source: 'measured' },
  { peer_id: 'so101-leader', joints: 6, robot_type: 'robot', role: 'leader', role_volts: 7.4, role_source: 'measured' },
  { peer_id: 'so101-real-689-twin', joints: 0, robot_type: 'sim' },
  { peer_id: 'so101-real-689-twin__so101', joints: 6, robot_type: 'sim' },
]

// ── twinFollowerOf: the joints live on the CHILD, not the twin process ──
{
  const { process, arm } = twinFollowerOf('so101-real-689', FLEET)
  assert.equal(process.peer_id, 'so101-real-689-twin')
  assert.equal(arm.peer_id, 'so101-real-689-twin__so101', 'the follower is the jointed child, never the process')
}
{
  const { process, arm } = twinFollowerOf('so101-leader', FLEET)
  assert.equal(process, null, 'no twin process → no follower')
  assert.equal(arm, null)
}

// ── the happy path: follower resolved, torque note surfaced honestly ──
{
  const plan = mirrorPlan('so101-real-689', FLEET)
  assert.equal(plan.blockers.length, 0)
  assert.equal(plan.follower, 'so101-real-689-twin__so101')
  assert.match(plan.notes.join(' '), /FOLLOWER \(12.6V\)/, 'a torqued arm is named as one')
  assert.match(plan.notes.join(' '), /not from this screen/, 'relaxing torque is the operator’s physical act, never the dashboard’s')
  assert.match(mirrorSentence(plan), /joint for joint/)
  assert.match(mirrorSentence(plan), /nothing physical moves/)
}

// ── a measured leader is hand-movable by design ──
{
  const fleet = FLEET.map(p => p.peer_id === 'so101-leader' ? p : p)
    .concat([{ peer_id: 'so101-leader-twin', joints: 0, robot_type: 'sim' },
             { peer_id: 'so101-leader-twin__so101', joints: 6, robot_type: 'sim' }])
  const plan = mirrorPlan('so101-leader', fleet)
  assert.equal(plan.blockers.length, 0)
  assert.match(plan.notes.join(' '), /hand-movable by design/)
}

// ── blockers, each naming its own remedy ──
{
  const plan = mirrorPlan('so101-leader', FLEET) // no twin spawned
  assert.equal(plan.follower, null)
  assert.match(mirrorSentence(plan), /spawn it first/)
}
{
  // twin up, child jointless (still loading): distinct sentence, not "no twin"
  const fleet = [
    { peer_id: 'arm-1', joints: 6, robot_type: 'robot' },
    { peer_id: 'arm-1-twin', joints: 0, robot_type: 'sim' },
    { peer_id: 'arm-1-twin__so101', joints: 0, robot_type: 'sim' },
  ]
  const plan = mirrorPlan('arm-1', fleet)
  assert.match(plan.blockers.join(' '), /still be loading/)
}
{
  // the jointless-arm state both real arms were in for three days
  const fleet = [
    { peer_id: 'arm-1', joints: 0, robot_type: 'robot' },
    { peer_id: 'arm-1-twin', joints: 0, robot_type: 'sim' },
    { peer_id: 'arm-1-twin__so101', joints: 6, robot_type: 'sim' },
  ]
  const plan = mirrorPlan('arm-1', fleet)
  assert.match(plan.blockers.join(' '), /no position to publish/)
  assert.match(plan.blockers.join(' '), /devices › logs/, 'the refusal points at where the reason lives')
}
{
  const plan = mirrorPlan('so101-real-689-twin__so101', FLEET)
  assert.match(plan.blockers.join(' '), /already a simulation/)
}
{
  const plan = mirrorPlan('ghost-arm', FLEET)
  assert.match(plan.blockers.join(' '), /not on the mesh/)
}

// ── joint-shape mismatch is a note, not a refusal (mesh maps by name) ──
{
  const fleet = [
    { peer_id: 'arm-1', joints: 6, robot_type: 'robot' },
    { peer_id: 'arm-1-twin', joints: 0, robot_type: 'sim' },
    { peer_id: 'arm-1-twin__so100', joints: 5, robot_type: 'sim' },
  ]
  const plan = mirrorPlan('arm-1', fleet)
  assert.equal(plan.blockers.length, 0)
  assert.match(plan.notes.join(' '), /only the names they share/)
}

// ── null/empty fleet never throws ──
assert.equal(mirrorPlan('x', null).follower, null)
assert.equal(mirrorPlan('x', []).follower, null)

console.log('mirrorToSim: all assertions passed')
