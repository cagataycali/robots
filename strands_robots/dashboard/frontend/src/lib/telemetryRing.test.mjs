// Run: node scripts/run-lib-tests.mjs telemetryRing
//
// This is the dashboard's only judgment about PHYSICAL MOTION, and statusSentence turns each of its
// three answers into a different accusation: `moving === false` accuses a running policy of being
// wedged, `moving === true` warns that an arm is moving with nobody commanding it, and null keeps
// quiet. Until now the logic lived inside useTelemetry's body, so it could only be exercised by
// rendering a component — nothing tested it.
//
// The cases below are the real fleet, not invented ones (measured 2026-08-21): both of cagatay's REAL
// arms publish NO `joints` key at all (two different root causes — a busy serial port and a missing
// calibration file), while only the sim twin so101-follower-twin__so101 publishes 6 joints. So "a peer
// with no joints" is the COMMON case here, and reporting its silence as stillness is what once made a
// card say "no joint data on this peer" and "idle and still — safe to approach" at the same time.
import assert from 'node:assert/strict'

const mod = await import('/tmp/telemetryRing.mjs')
const { advance, emptyRing, summarize, jointValues, motionBetween, TELEMETRY_CAP } = mod

/** Fold a sequence of frames in, one state timestamp per frame, 10 Hz like the real topic. */
const drive = (frames, t0 = 1_000) => {
  let acc = emptyRing()
  frames.forEach((joints, i) => {
    const peer = { state: joints === null ? { t: t0 + i * 0.1 } : { t: t0 + i * 0.1, joints } }
    acc = advance(acc, peer, t0 + i * 0.1)
  })
  return acc
}

// ── 1. joint payload shapes: whatever the robot published, the same vector comes out ──
assert.deepEqual(jointValues({ state: { joints: { a: 1, b: 2 } } }), [1, 2], 'bare numbers')
assert.deepEqual(jointValues({ state: { joints: { a: [3, 99] } } }), [3], 'a [position, velocity] pair uses position')
assert.deepEqual(jointValues({ state: { joints: { a: { position: 4 } } } }), [4], 'an object uses .position')
assert.deepEqual(jointValues({ state: { joints: {} } }), [], 'an empty joints map is no joints')
assert.deepEqual(jointValues({ state: { t: 1 } }), [], 'NO joints key — both real arms today')
assert.deepEqual(jointValues({}), [], 'no state at all does not throw')

// ── 2. a reshaped vector must not manufacture a motion spike ──
assert.equal(motionBetween([0, 0], [1, 1]), 1, 'mean absolute change per joint')
assert.equal(motionBetween([], [1, 2, 3]), 0, 'the first frame has nothing to difference against')
assert.equal(motionBetween([0, 0], [0, 0, 0]), 0, 'a joint COUNT change reports no motion, not a spike')

// ── 3. THE LAW: a peer publishing no joints gets NO OPINION, ever ──
// 20 frames of presence/state with no joints key — the live state of so101-follower and so101-leader.
const noJoints = summarize(drive(Array(20).fill(null)), 1_002)
assert.equal(noJoints.jointsSeen, false, 'we heard state frames and none carried joints')
assert.equal(noJoints.moving, null,
             'NO JOINTS MEANS NO OPINION: motion is computed FROM joints, so 20 zeros are an empty ' +
             'stream, not a measurement. Answering false here is how a card said "no joint data" and ' +
             '"idle and still — safe to approach" at once.')
assert.equal(noJoints.samples.length, 20, 'the frames are still counted — the peer IS publishing')
assert.ok(noJoints.hz > 9 && noJoints.hz < 11, `rate is measured without joints too, got ${noJoints.hz}`)

// ── 4. before anything is heard, "no joints yet" must not look like "never any joints" ──
const nothing = summarize(emptyRing(), 1_000)
assert.equal(nothing.jointsSeen, null, 'null = we have not heard a single state frame')
assert.equal(nothing.moving, null, 'and therefore no motion opinion')
assert.equal(nothing.stateAgeS, null, 'an age of 0 would claim a sample that never arrived')
assert.equal(nothing.hz, 0, 'no rate from no samples')

// A peer that published joints once and then dropped them has not stopped being an arm.
const dropped = summarize(drive([{ a: 1 }, { a: 1 }, null, null, null]), 1_001)
assert.equal(dropped.jointsSeen, true, 'jointsSeen latches: seen once, seen thereafter')

// ── 5. measured stillness IS allowed — when there is something to measure ──
const still = summarize(drive(Array(15).fill({ shoulder_pan: 0.5, elbow_flex: -0.25 })), 1_002)
assert.equal(still.jointsSeen, true, 'joints are present')
assert.equal(still.moving, false,
             'identical positions with joints present is a real measurement of stillness — this is the ' +
             'evidence statusSentence needs to call a running policy wedged')

// ── 6. an arm actually moving says so ──
const sweeping = summarize(drive(Array(15).fill(0).map((_, i) => ({ shoulder_pan: i * 2.5 }))), 1_002)
assert.equal(sweeping.moving, true, 'a joint marching 2.5 deg per frame is moving')

// Fewer than 10 samples is not enough to accuse anyone.
assert.equal(summarize(drive(Array(9).fill(0).map((_, i) => ({ a: i }))), 1_001).moving, null,
             'an accusation off 9 samples is noise — null until the ring has 10')

// ── 7. a repeated state timestamp is not a new sample (the hook's re-render guard) ──
let acc = emptyRing()
const frame = { state: { t: 7, joints: { a: 1 } } }
acc = advance(acc, frame, 1_000)
const same = advance(acc, frame, 1_001)
assert.equal(same, acc, 'advance returns the SAME object for an already-folded timestamp, so the hook cannot loop')
assert.equal(advance(acc, { state: { joints: { a: 1 } } }, 1_001), acc, 'a frame with no timestamp is not folded')

// ── 8. the ring is bounded: 12s of history, never a leak on a peer left open for hours ──
const long = drive(Array(TELEMETRY_CAP + 40).fill({ a: 1 }))
assert.equal(long.samples.length, TELEMETRY_CAP, `capped at ${TELEMETRY_CAP}`)
assert.ok(long.samples[0].t > 1_003, 'the OLDEST samples are the ones dropped')

// ── 9. PINNED, NOT ENDORSED: the threshold is purely relative, so encoder dither reads as motion ──
// `moving` compares recent motion to the ring's own PEAK (5%), with no absolute floor beyond a
// divide-by-zero guard. That is deliberate — motion carries the joints' unit (the real arms report
// DEGREES, the sim twin radians), and an absolute epsilon in the wrong unit is the Q27 bug class. But
// it means a torqued, commanded-still arm whose encoder dithers by a hair reports moving: true, which
// statusSentence renders as an uncommanded-motion warning. Filed as Q85; this assertion documents
// today's answer so the fix is a deliberate change with a failing test, not a silent drift.
const dither = summarize(drive(Array(15).fill(0).map((_, i) => ({ a: 0.5 + (i % 2) * 0.0004 }))), 1_002)
assert.equal(dither.moving, true, 'Q85: sub-milli-unit dither currently reads as motion (pinned, not endorsed)')

console.log('telemetryRing.test.mjs: all assertions passed')
