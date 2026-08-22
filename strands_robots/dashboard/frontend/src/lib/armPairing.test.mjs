// Run: npx esbuild src/lib/armPairing.ts --bundle --format=esm --outfile=/tmp/armPairing.mjs
// && node src/lib/armPairing.test.mjs THE BUG THIS FILE EXISTS FOR: RecordPanel defaulted the
// teleop pair with /leader|arm-2/. arm-2 measures 12.6V — it IS the follower — so the record
// screen pre-filled the pair BACKWARDS, and a backwards pair means the operator hand-forces a
// torqued 12V arm while the 7.4V one tries to mirror it, and the dataset teaches the wrong
// arm.
import assert from 'node:assert/strict'
import { pairArms, measured, roleLabel, contradiction } from '/tmp/armPairing.mjs'

const arm = (peer_id, role, role_volts) => ({ peer_id, role, role_volts })

// ── measurement decides ──
const both = pairArms([arm('so101-arm-1', 'leader', 7.4), arm('so101-arm-2', 'follower', 12.6)])
assert.deepEqual(both, { leader: 'so101-arm-1', follower: 'so101-arm-2', basis: 'measured' })
// the ORIGINAL inversion: the name index says arm-2 is the leader, the bus says otherwise
const inverted = pairArms([arm('arm-1', 'follower', 12.6), arm('arm-2', 'leader', 7.4)])
assert.equal(inverted.leader, 'arm-2', 'the measurement beats the index in the name')
assert.equal(inverted.follower, 'arm-1')
// a measurement also beats a name that CONTRADICTS it
const lying = pairArms([arm('so101-leader', 'follower', 12.6), arm('so101-follower', 'leader', 7.4)])
assert.equal(lying.leader, 'so101-follower')
assert.equal(lying.basis, 'measured')
assert.deepEqual(measured([arm('a', 'leader'), arm('b', 'follower'), arm('c', null)], 'leader').map(c => c.peer_id), ['a'])

const onlyLeader = pairArms([arm('a', 'leader', 7.4), arm('b', null)])
assert.equal(onlyLeader.leader, 'a')
assert.equal(onlyLeader.follower, '', 'the unmeasured arm is NOT promoted by elimination')
assert.match(onlyLeader.note, /no arm has measured as a 12V follower/)
const onlyFollower = pairArms([arm('a', 'follower', 12.6), arm('b', null)])
assert.equal(onlyFollower.follower, 'a')
assert.equal(onlyFollower.leader, '')
assert.match(onlyFollower.note, /5.5V on the USB rail/, 'says why a leader may be missing')

const oneLeaderTwoFollowers = pairArms([
  arm('a', 'leader', 7.4), arm('b', 'follower', 12.6), arm('c', 'follower', 12.4),
])
assert.equal(oneLeaderTwoFollowers.leader, 'a', 'the certain slot is kept')
assert.equal(oneLeaderTwoFollowers.follower, '')
assert.equal(oneLeaderTwoFollowers.basis, 'measured')
assert.match(oneLeaderTwoFollowers.note, /2 arms measured as followers/)
const oneFollowerTwoLeaders = pairArms([
  arm('a', 'follower', 12.6), arm('b', 'leader', 7.4), arm('c', 'leader', 7.6),
])
assert.equal(oneFollowerTwoLeaders.follower, 'a')
assert.equal(oneFollowerTwoLeaders.leader, '')
assert.match(oneFollowerTwoLeaders.note, /2 arms measured as leaders/)

// genuinely ambiguous on both sides (or one side ambiguous, the other silent): pick nothing, explain
const twoFollowers = pairArms([arm('a', 'follower', 12.6), arm('b', 'follower', 12.5)])
assert.deepEqual([twoFollowers.leader, twoFollowers.follower, twoFollowers.basis], ['', '', 'none'])
assert.match(twoFollowers.note, /2 arms measured as the follower/)
assert.match(twoFollowers.note, /check the volts/, 'points at the evidence the human can read')
const messy = pairArms([arm('a', 'leader'), arm('b', 'leader'), arm('c', 'follower'), arm('d', 'follower')])
assert.equal(messy.basis, 'none')
assert.match(messy.note, /2 arms measured as the leader and 2 as the follower/)

const named = pairArms([arm('so101-leader', null), arm('so101-follower', null)])
assert.deepEqual([named.leader, named.follower, named.basis], ['so101-leader', 'so101-follower', 'named'])
assert.match(named.note, /taken at their word/, 'the weaker basis is stated, not hidden')
// AN INDEX IN A NAME IS NEVER EVIDENCE — the whole original bug
const indexed = pairArms([arm('arm-1', null), arm('arm-2', null)])
assert.deepEqual([indexed.leader, indexed.follower, indexed.basis], ['', '', 'none'])
assert.match(indexed.note, /the leader is the lighter 7.4V arm/)
// only one name states a role: fill that slot, leave the other to the human (elimination is an
// inference by another route, and it is exactly what this file refuses)
const halfNamed = pairArms([arm('so101-leader', null), arm('arm-2', null)])
assert.equal(halfNamed.leader, 'so101-leader')
assert.equal(halfNamed.follower, '')
assert.equal(halfNamed.basis, 'named')
assert.match(halfNamed.note, /the other arm is left to you/)
// "leader" must be a WORD, not a substring: leaderboard-arm states nothing
assert.equal(pairArms([arm('leaderboard-arm', null), arm('x', null)]).basis, 'none')
assert.equal(pairArms([arm('SO101-LEADER', null), arm('SO101-FOLLOWER', null)]).basis, 'named', 'case-insensitive')
// one peer named both things cannot be both slots
const contradictory = pairArms([arm('leader-follower-arm', null)])
assert.equal(contradictory.basis, 'none')
// no arms at all says so, rather than blaming the measurement
assert.equal(pairArms([]).note, 'no arms on the mesh')

// ── labels carry the volts, so the evidence travels with the name ──
assert.equal(roleLabel(arm('a', 'follower', 12.6)), 'a — follower · 12.6V')
assert.equal(roleLabel(arm('a', 'leader', null)), 'a — leader')
assert.equal(roleLabel(arm('a', null)), 'a — role not measured')
assert.equal(roleLabel(arm('a', 'unpowered', 5.5)), 'a — unpowered · 5.5V')

// ── the contradiction check: a warning, with the PHYSICAL consequence ──
const fleet = [arm('a', 'leader', 7.4), arm('b', 'follower', 12.6), arm('c', null), arm('d', 'unpowered', 5.5), arm('e', 'mixed', 9.1)]
assert.equal(contradiction(fleet, 'leader', 'a'), null, 'agreement is silent')
assert.equal(contradiction(fleet, 'follower', 'b'), null)
assert.equal(contradiction(fleet, 'leader', ''), null, 'an empty slot is not a contradiction')
assert.equal(contradiction(fleet, 'leader', 'c'), null, 'an UNMEASURED arm is never a warning — we do not know')
assert.equal(contradiction(fleet, 'leader', 'ghost'), null, 'an arm not on the mesh is not a role claim')
const wrongWay = contradiction(fleet, 'leader', 'b')
assert.match(wrongWay, /12.6V/)
assert.match(wrongWay, /hand-moving a torqued 12V arm/, 'names what would physically happen')
assert.match(contradiction(fleet, 'follower', 'a'), /should be the 12V one that mirrors your hand/)
assert.match(contradiction(fleet, 'leader', 'd'), /power supply is off/, 'unpowered is a supply fault, not a role')
assert.match(contradiction(fleet, 'follower', 'd'), /cannot hold or mirror anything/)
assert.match(contradiction(fleet, 'leader', 'e'), /wiring fault/, 'mixed volts is a fault, not a role')

console.log('armPairing: all assertions passed')
