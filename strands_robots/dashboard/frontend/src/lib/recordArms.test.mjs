import assert from 'node:assert/strict'
import { armJointWarning } from '/tmp/recordArms.mjs'

const NOW = 1_000_000
const peer = (joints, { age = 2, problem = null, presence = { hw: 'so_follower' } } = {}) => ({
  peer_id: 'so101-leader', last_seen: NOW - age, presence, joint_problem: problem,
  state: joints === undefined ? undefined : { peer_id: 'so101-leader', t: NOW - age, joints },
})

// --- it warns when there IS evidence ---------------------------------------
{
  const w = armJointWarning(peer(null), { slot: 'follower', nowS: NOW })
  assert.match(w, /reports no joint positions/)
  assert.match(w, /no observations to learn from/, "the follower's joints are the observations")
  assert.match(w, /will be refused/, 'the server will refuse it, so say so rather than implying a choice')
}
{
  const w = armJointWarning(peer({}), { slot: 'leader', nowS: NOW })
  assert.match(w, /no actions to learn from/, "the leader's joints are the actions")
}
// The REASON is the card's sentence, not a second vocabulary for the same fault.
{
  const w = armJointWarning(peer(null, {
    problem: { headline: 'this board has no calibration', remedy: 'Respawn it as leader_arm.' },
  }), { slot: 'follower', nowS: NOW })
  assert.match(w, /this board has no calibration/)
  assert.match(w, /Respawn it as leader_arm\./)
}

// --- and stays quiet without evidence, exactly like the server -------------
assert.equal(armJointWarning(peer({ 'shoulder_pan.pos': 1 }), { slot: 'follower', nowS: NOW }), null)
assert.equal(armJointWarning(null, { slot: 'follower', nowS: NOW }), null)
assert.equal(armJointWarning(undefined, { slot: 'leader', nowS: NOW }), null)
assert.equal(armJointWarning(peer(undefined), { slot: 'follower', nowS: NOW }), null,
  'no state document at all is the ageing gate\'s business, not this one\'s')
assert.equal(armJointWarning(peer('six'), { slot: 'follower', nowS: NOW }), null,
  'a joints shape we do not understand is not absence')
assert.equal(armJointWarning(peer(null, { age: 31 }), { slot: 'follower', nowS: NOW }), null,
  'stale silence is not evidence about now (the pre-Q80 trap)')
assert.ok(armJointWarning(peer(null, { age: 29 }), { slot: 'follower', nowS: NOW }),
  'and the boundary is where the server puts it')
{
  const p = peer(null)
  delete p.last_seen
  assert.equal(armJointWarning(p, { slot: 'follower', nowS: NOW }), null,
    'an undateable reading cannot be called fresh')
}
console.log('recordArms: all assertions passed')
