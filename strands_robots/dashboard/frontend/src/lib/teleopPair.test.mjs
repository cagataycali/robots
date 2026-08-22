import assert from 'node:assert/strict'
import { leaderOptions, pairPlan, pairSentence } from '/tmp/teleopPair.mjs'

// This fleet, as it actually is today: two real arms reporting NO joints, a simulator process with 0
// joints, and the robot under it with 6.
const FLEET = [
  { peer_id: 'so101-follower', joints: 0, role: 'follower', role_volts: 12.6, role_source: 'measured' },
  { peer_id: 'so101-leader', joints: 0, role: null, role_source: null },
  { peer_id: 'so101-follower-twin', joints: 0 },
  { peer_id: 'so101-follower-twin__so101', joints: 6 },
]

const opts = leaderOptions('so101-follower', FLEET)
assert.equal(opts.find(o => o.peer_id === 'so101-follower'), undefined, 'an arm is never offered as its own leader')
const twinHost = opts.find(o => o.peer_id === 'so101-follower-twin')
assert.equal(twinHost.ok, false); assert.match(twinHost.why, /this is the process/, 'a process is not an arm')
const jointless = opts.find(o => o.peer_id === 'so101-leader')
assert.equal(jointless.ok, false)
assert.match(jointless.why, /cannot publish a position it cannot read/)
assert.match(jointless.why, /devices › logs/, 'the refusal points at where the reason lives')
const real = opts.find(o => o.peer_id === 'so101-follower-twin__so101')
assert.equal(real.ok, true, 'the peer that actually reports joints is offerable')
assert.match(real.why, /role not measured/, 'unmeasured role is stated, not guessed')

// A parent/child pair is the SAME robot: never offered as its own leader.
const childOpts = leaderOptions('so101-follower-twin__so101', FLEET)
assert.equal(childOpts.find(o => o.peer_id === 'so101-follower-twin').ok, false)

const measured = leaderOptions('a', [
  { peer_id: 'a', joints: 6 },
  { peer_id: 'lead', joints: 6, role: 'leader', role_volts: 7.4, role_source: 'measured' },
  { peer_id: 'foll', joints: 6, role: 'follower', role_volts: 12.6, role_source: 'measured' },
])
assert.match(measured.find(o => o.peer_id === 'lead').why, /measured as a leader \(7\.4V\)/)
const followerAsLeader = measured.find(o => o.peer_id === 'foll')
assert.equal(followerAsLeader.ok, true, 'a follower-wired arm may still be hand-guided — evidence, not a veto')
assert.match(followerAsLeader.why, /measured as a FOLLOWER \(12\.6V\)/)

// THE PLAN.
const plan = pairPlan('f', 'l', [{ peer_id: 'f', joints: 6 }, { peer_id: 'l', joints: 6 }])
assert.deepEqual(plan.consents, ['agent_physical_motion', 'teleop_degree_units'])
assert.deepEqual(plan.blockers, [])
assert.equal(pairSentence(plan), 'f could follow l')

// A jointless arm on either end is a hard blocker, and each blocker NAMES the peer.
const dead = pairPlan('so101-follower', 'so101-leader', FLEET)
assert.equal(dead.blockers.length, 2)
assert.match(dead.blockers[0], /so101-follower reports no joints, so nothing could be applied/)
assert.match(dead.blockers[1], /so101-leader reports no joints, so it has no position to publish/)
assert.match(pairSentence(dead), /^cannot start: /)

// An absent peer cannot be commanded, and a host process is refused by name.
const gone = pairPlan('ghost', 'l', [{ peer_id: 'l', joints: 6 }])
assert.match(gone.blockers[0], /ghost is not on the mesh/)
const hosted = pairPlan('so101-follower-twin', 'so101-follower-twin__so101', FLEET)
assert.ok(hosted.blockers.some(b => /so101-follower-twin hosts/.test(b)), 'the process must not be the follower')

const shapes = pairPlan('f', 'l', [
  { peer_id: 'f', joints: 5, role: 'leader', role_volts: 7.4, role_source: 'measured' },
  { peer_id: 'l', joints: 6 }])
assert.deepEqual(shapes.blockers, [], 'a shape mismatch does not block — only the shared names are followed')
assert.equal(shapes.notes.length, 2)
assert.ok(shapes.notes.some(n => /only the names they share/.test(n)))
assert.ok(shapes.notes.some(n => /is about to be DRIVEN/.test(n)))
assert.match(pairSentence(shapes), /f could follow l · /)

// Degenerate asks produce nothing rather than a guess.
for (const bad of [['', 'l'], ['f', ''], ['f', 'f']]) assert.equal(pairPlan(bad[0], bad[1], FLEET), null)
assert.equal(pairSentence(null), null)
console.log('teleopPair: all assertions passed')
