import assert from 'node:assert/strict'
import { humanJointName, humanJointNames, stripLegend } from '/tmp/jointLabels.mjs'

// --- reformatting only, never translating ----------------------------------
assert.equal(humanJointName('shoulder_pan'), 'Shoulder pan')
assert.equal(humanJointName('shoulder_lift.pos'), 'Shoulder lift')
assert.equal(humanJointName('wrist_roll_pos'), 'Wrist roll')
assert.equal(humanJointName('gripper'), 'Gripper')
assert.equal(humanJointName('left_hip_yaw'), 'Left hip yaw')
// the words themselves are never rewritten: 'flex' stays 'flex', so the label
// can always be matched back to the key an operator types in a config
for (const k of ['elbow_flex', 'wrist_flex', 'ankle_pitch']) {
  const words = k.split('_')
  for (const w of words) assert.ok(humanJointName(k).toLowerCase().includes(w), `${k} lost ${w}`)
}
// degenerate input never throws and never renders empty
for (const k of ['', '_', '.pos', '_pos', '   ']) {
  const l = humanJointName(k)
  assert.equal(typeof l, 'string')
  assert.ok(l.length > 0 || k.length === 0, JSON.stringify(k))
}
// idempotent-ish: humanising an already-human label does not degrade it
assert.equal(humanJointName(humanJointName('shoulder_pan')), 'Shoulder pan')

// --- a label may never collide ---------------------------------------------
const so101 = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
assert.deepEqual(humanJointNames(so101),
  ['Shoulder pan', 'Shoulder lift', 'Elbow flex', 'Wrist flex', 'Wrist roll', 'Gripper'])
// two keys that would read the same => the whole strip falls back to raw keys
const colliding = ['wrist_flex', 'wrist.flex', 'gripper']
assert.deepEqual(humanJointNames(colliding), colliding, 'ambiguous labels must not be shown')
const alsoColliding = ['arm_1.pos', 'arm_1_pos']
assert.deepEqual(humanJointNames(alsoColliding), alsoColliding)
assert.deepEqual(humanJointNames([]), [])
// same length, same order, always — the strip zips these against values
for (const set of [so101, colliding, ['a'], []]) {
  assert.equal(humanJointNames(set).length, set.length)
}

// --- the unit is stated once, and honestly ---------------------------------
const degGrip = stripLegend('servo', 60000, so101)
assert.match(degGrip, /degrees/)
assert.match(degGrip, /gripper on its own/, 'the one row whose unit differs must be called out')
assert.match(degGrip, /last 60s/)
assert.match(degGrip, /bar =/)
// no gripper in the stream => no claim about one
const degOnly = stripLegend('servo', 60000, ['shoulder_pan', 'elbow_flex'])
assert.match(degOnly, /degrees/)
assert.doesNotMatch(degOnly, /gripper/)
// radians are never called degrees
const rad = stripLegend('radian', 30000, so101)
assert.match(rad, /radians/)
assert.doesNotMatch(rad, /degrees/)
assert.match(rad, /last 30s/)
// a per-row '°' would lie on the gripper row, so the legend must not promise one
assert.doesNotMatch(degGrip, /°/)

console.log('jointLabels: all assertions passed')
