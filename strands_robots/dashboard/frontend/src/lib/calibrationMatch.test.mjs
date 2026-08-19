import assert from 'node:assert/strict'
import { calibrationVerdict } from '/tmp/calibrationMatch.mjs'

// --- THE DEFECT ------------------------------------------------------------
// The spawn form's "Calibration id" was free text under a prose warning that
// nothing enforced. A typo spawns happily and the arm runs on raw servo counts
// with the wrong joint limits - a physical outcome, discovered by watching a
// real arm reach somewhere it should not be able to reach.

const E = (id, model = 'so101_follower', extra = {}) => ({
  deviceType: 'robots', model, id, unreadable: false, ...extra,
})
const LIST = [E('follower_arm'), E('leader_arm', 'so101_leader')]

// An exact id is confirmed, with the evidence for the confirmation.
{
  const v = calibrationVerdict('follower_arm', [E('follower_arm', 'so101_follower', { motors: 6 })], 'so101')
  assert.equal(v.kind, 'match')
  assert.equal(v.warn, false)
  assert.match(v.note, /matches follower_arm \(so101_follower, 6 motors\)/)
}

// THE TYPO: underscore vs hyphen. This is the case that actually happens.
{
  const v = calibrationVerdict('follower-arm', LIST, 'so101')
  assert.equal(v.kind, 'suggest')
  assert.equal(v.warn, true)
  assert.equal(v.suggestion, 'follower_arm')
  assert.match(v.note, /did you mean follower_arm/)
}

// A name nobody has: warn AND list what exists, so the next attempt can work.
{
  const v = calibrationVerdict('gripper9000', LIST, 'so101')
  assert.equal(v.kind, 'unknown')
  assert.equal(v.warn, true)
  assert.match(v.note, /follower_arm, leader_arm/)
  assert.match(v.note, /raw servo counts/)
}

// Same name, wrong robot: the silent one. The limits would come from another arm.
{
  const v = calibrationVerdict('follower_arm', [E('follower_arm', 'so100_follower')], 'so101')
  assert.equal(v.warn, true)
  assert.match(v.note, /calibrated for so100_follower, not so101/)
}

// ...but a family that is merely written differently is not an accusation.
for (const [model, family] of [['so101_follower', 'so101'], ['so101', 'so101_follower'], ['so101_leader', 'SO101']]) {
  assert.equal(calibrationVerdict('x', [E('x', model)], family).warn, false, `${model} vs ${family}`)
}

// An id that exists but whose FILE is unreadable must not read as a green match.
{
  const v = calibrationVerdict('follower_arm', [E('follower_arm', 'so101_follower', { unreadable: true })], 'so101')
  assert.equal(v.warn, true)
  assert.match(v.note, /could not be read/)
}

// Empty id: the honest default is "uncalibrated", and the real names help.
{
  const v = calibrationVerdict('', LIST, 'so101')
  assert.equal(v.kind, 'none')
  assert.equal(v.warn, true)
  assert.match(v.note, /raw servo counts/)
  assert.match(v.note, /follower_arm, leader_arm/)
}

// No files at all: say that, rather than "did you mean" nothing.
{
  const v = calibrationVerdict('follower_arm', [], 'so101')
  assert.equal(v.kind, 'unknown')
  assert.match(v.note, /no calibration files exist on this machine/)
  assert.match(v.note, /spawn now and calibrate after/, 'must not read like a refusal')
}

// The list has not arrived / the API failed: SAY NOTHING rather than guess. A
// guess would either accuse a correct id or bless a wrong one.
for (const missing of [null, undefined]) {
  const v = calibrationVerdict('follower_arm', missing, 'so101')
  assert.equal(v.kind, 'unchecked')
  assert.equal(v.warn, false)
  assert.match(v.note, /not checked/)
  assert.equal(calibrationVerdict('', missing).note, '', 'nothing typed, nothing to say')
}

// Whitespace and case are not new ids.
for (const typed of ['  follower_arm  ', 'FOLLOWER_ARM', 'Follower_Arm']) {
  assert.equal(calibrationVerdict(typed, LIST, 'so101').kind, 'match', typed)
}

// Entries with no id (an unparsed row) must not become a phantom match.
{
  const v = calibrationVerdict('follower_arm', [E(''), E('  ')], 'so101')
  assert.equal(v.kind, 'unknown')
  assert.match(v.note, /no calibration files exist/)
}

console.log('calibrationMatch: all assertions passed')
