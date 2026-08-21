// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
//
// THIS MODULE DECIDES WHAT A JOINT BAR MEANS. It replaced a per-joint rule (`|pos| > 4 ? 100 : PI`)
// that put two joints of one arm on different axes and flipped an axis mid-stream; the whole point of
// the replacement is that the axis is a property of the STREAM and changes only under hysteresis. None
// of that was pinned by a test until now, so nothing stopped a later edit from restoring the old
// behaviour with a one-line "simplification".
import assert from 'node:assert/strict'
import {
  decideStripScale, fillPercent, frameEvidence, defaultSpan, isOneSidedJoint,
  RADIAN_CEILING, RADIAN_FLOOR, SWITCH_FRAMES,
} from '/tmp/jointScale.mjs'

// ── the evidence a single frame gives, and when it must stay silent ──
// cagatay's SO-101 reports DEGREES (wrist_roll rests at 170, shoulder_lift near -46) — the case Q27
// was about. Degrees are the 'servo' side of this taxonomy.
assert.equal(frameEvidence([['wrist_roll.pos', 170], ['shoulder_lift.pos', -46]]), 'servo')
assert.equal(frameEvidence([['elbow_flex.pos', 0.7], ['wrist_flex.pos', -1.2]]), 'radian')
// AMBIGUOUS BAND: between the floor and the ceiling either unit is possible, so the frame has NO
// opinion. Reporting one here is how a hovering stream oscillates.
assert.equal(frameEvidence([['j', (RADIAN_FLOOR + RADIAN_CEILING) / 2]]), undefined)
// an all-zero frame fits both units and must never count as evidence
assert.equal(frameEvidence([['a', 0], ['b', 0]]), undefined, 'zeros are not evidence')
assert.equal(frameEvidence([]), undefined)
assert.equal(frameEvidence([['a', NaN], ['b', Infinity]]), undefined, 'no finite sample = no opinion')
// one finite sample among junk still speaks
assert.equal(frameEvidence([['a', NaN], ['b', 88]]), 'servo')
// the peak decides, not the average: one degree-scaled joint makes the STREAM degree-scaled
assert.equal(frameEvidence([['a', 0.1], ['b', 0.2], ['c', 91]]), 'servo')

// ── one-sided joints are known by NAME, deliberately: a stable property, not the live value ──
for (const n of ['gripper', 'gripper.pos', 'left_grip', 'jaw', 'finger_1', 'claw']) {
  assert.ok(isOneSidedJoint(n), n)
}
for (const n of ['shoulder_pan', 'elbow_flex', 'wrist_roll']) assert.equal(isOneSidedJoint(n), false, n)
assert.deepEqual(defaultSpan('gripper', 'servo'), { lo: 0, hi: 100 }, 'closed..open, not -100..100')
assert.deepEqual(defaultSpan('elbow_flex', 'servo'), { lo: -100, hi: 100 })
assert.deepEqual(defaultSpan('gripper', 'radian'), { lo: -Math.PI, hi: Math.PI },
  'radian streams put every joint on one axis, gripper included')

// ── ONE AXIS PER STRIP: the defect that started this module ──
// A degree-scaled gripper at 45 and an elbow at 0.7 on the same card must share a unit.
const mixed = decideStripScale([['gripper', 45], ['elbow_flex', 0.7]])
assert.equal(mixed.unit, 'servo', 'the first frame may pick outright — no axis exists to protect yet')
assert.deepEqual(mixed.ranges.gripper, { lo: 0, hi: 100 })
assert.deepEqual(mixed.ranges.elbow_flex, { lo: -100, hi: 100 },
  'the elbow is scaled in the STRIP\'s unit, not in the one its own small value suggests')

// ── HYSTERESIS: a glitch frame cannot flip the axis ──
let memo = decideStripScale([['a', 0.5]])
assert.equal(memo.unit, 'radian')
memo = decideStripScale([['a', 99]], memo)          // one wild frame
assert.equal(memo.unit, 'radian', 'a single frame never flips the axis')
assert.equal(memo.pending, 'servo')
assert.equal(memo.pendingFrames, 1)
memo = decideStripScale([['a', 0.5]], memo)         // back to normal
assert.equal(memo.pending, null, 'an agreeing frame breaks the streak')
assert.equal(memo.pendingFrames, 0)

// a SUSTAINED argument does switch — on exactly the documented frame, not before
memo = decideStripScale([['a', 0.5]])
for (let i = 1; i < SWITCH_FRAMES; i++) {
  memo = decideStripScale([['a', 99]], memo)
  assert.equal(memo.unit, 'radian', `still radian after ${i} disagreeing frame(s)`)
  assert.equal(memo.pendingFrames, i)
}
memo = decideStripScale([['a', 99]], memo)
assert.equal(memo.unit, 'servo', `switched on frame ${SWITCH_FRAMES}`)
assert.equal(memo.pending, null)
assert.equal(memo.pendingFrames, 0)

// ── a unit change DISCARDS observed extremes: they were measured on the other axis ──
let m2 = decideStripScale([['a', 3.0]])
m2 = decideStripScale([['a', -3.4]], m2)
assert.equal(m2.ranges.a.lo, -3.4, 'observation widens a range')
for (let i = 0; i < SWITCH_FRAMES; i++) m2 = decideStripScale([['a', 120]], m2)
assert.equal(m2.unit, 'servo')
assert.deepEqual(m2.ranges.a, { lo: -100, hi: 120 },
  'the radian extreme is gone: carrying it over would keep the old scale alive inside the new one')

// widening NEVER shrinks, so a bar cannot re-scale under a still arm
let m3 = decideStripScale([['a', 0.1]])
m3 = decideStripScale([['a', 5.5]], m3)   // one ambiguous-to-servo excursion, no switch yet
const wide = m3.ranges.a.hi
m3 = decideStripScale([['a', 0.1]], m3)
assert.equal(m3.ranges.a.hi, wide, 'a range that has been widened stays widened')

// ── an ambiguous frame changes NOTHING, counter included ──
// (a zero frame from a robot between reads, or a pass through the ambiguous band, must not wipe the
// evidence collected so far — that is how a degree stream used to stall on the radian axis forever)
let m7 = decideStripScale([['a', 0.5]])
m7 = decideStripScale([['a', 99]], m7)
assert.equal(m7.pendingFrames, 1)
m7 = decideStripScale([['a', 0]], m7)
assert.equal(m7.pendingFrames, 1, 'a zero frame neither advances nor resets the streak')
assert.equal(m7.pending, 'servo')
m7 = decideStripScale([['a', 3.6]], m7)   // inside the ambiguous band
assert.equal(m7.pendingFrames, 1, 'the ambiguous band is silence, not disagreement')
for (let i = 2; i <= SWITCH_FRAMES; i++) m7 = decideStripScale([['a', 99]], m7)
assert.equal(m7.unit, 'servo', 'interleaved silence still lets a sustained argument land')

// a joint that disappears from the stream takes its range with it (a range is per joint, per strip)
const m4 = decideStripScale([['b', 0.2]], m3)
assert.equal(m4.ranges.a, undefined)
assert.ok(m4.ranges.b)

// a non-finite sample must not poison a range — it is dropped, and 0 stands in for the bar
const m5 = decideStripScale([['a', NaN]])
assert.ok(Number.isFinite(m5.ranges.a.lo) && Number.isFinite(m5.ranges.a.hi))

// an empty frame keeps the established unit rather than reverting to the seed default
const m6 = decideStripScale([], { ...m2 })
assert.equal(m6.unit, 'servo', 'no samples is not evidence for the other axis')

// ── fillPercent: clamped, and never NaN into a CSS width ──
assert.equal(fillPercent(0, { lo: -100, hi: 100 }), 50)
assert.equal(fillPercent(-100, { lo: -100, hi: 100 }), 0)
assert.equal(fillPercent(100, { lo: -100, hi: 100 }), 100)
assert.equal(fillPercent(250, { lo: -100, hi: 100 }), 100, 'clamped: a bar cannot overflow its track')
assert.equal(fillPercent(-250, { lo: -100, hi: 100 }), 0)
assert.equal(fillPercent(5, { lo: 5, hi: 5 }), 0, 'a zero-width range yields 0, not a division by zero')
assert.equal(fillPercent(NaN, { lo: 0, hi: 1 }), 0, 'NaN never reaches a style attribute')
assert.equal(fillPercent(0.4, { lo: 0, hi: 100 }), 0.4)

console.log('jointScale: all assertions passed')
