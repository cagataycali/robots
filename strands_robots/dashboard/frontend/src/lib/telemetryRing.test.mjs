// Run: node scripts/run-lib-tests.mjs telemetryRing This is the dashboard's only judgment
// about PHYSICAL MOTION, and statusSentence turns each of its three answers into a different
// accusation: `moving === false` accuses a running policy of being wedged, `moving === true`
// warns that an arm is moving with nobody commanding it, and null keeps quiet.
import assert from 'node:assert/strict'

const mod = await import('/tmp/telemetryRing.mjs')
const { advance, emptyRing, summarize, jointValues, motionBetween, TELEMETRY_CAP, recentRun, TELEMETRY_GAP_S } = mod

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

// ── 9.
const dither = summarize(drive(Array(15).fill(0).map((_, i) => ({ a: 0.5 + (i % 2) * 0.0004 }))), 1_002)
assert.equal(dither.moving, true, 'Q85: sub-milli-unit dither currently reads as motion (pinned, not endorsed)')

const S = (t, motion) => ({ t, motion })

assert.deepEqual(recentRun([]), [], 'nothing is its own trailing run')
assert.deepEqual(recentRun([S(1, 0)]), [S(1, 0)], 'one sample is a run of one')
const dense = [S(1, 0), S(1.1, 0), S(1.2, 0)]
assert.deepEqual(recentRun(dense), dense, 'a stream with no gap is entirely current')
assert.deepEqual(recentRun([S(1, 5), S(2, 5), S(600, 1), S(600.1, 1)]), [S(600, 1), S(600.1, 1)],
                 'the run starts AFTER the silence')
assert.deepEqual(recentRun([S(1, 0), S(1 + TELEMETRY_GAP_S, 0)]).length, 2,
                 'a gap of exactly the threshold is not yet a break (> not >=)')
assert.deepEqual(recentRun([S(1, 0), S(1 + TELEMETRY_GAP_S + 0.01, 0)]).length, 1, 'just over it is')
assert.deepEqual(recentRun([S(1, 0), S(50, 0), S(50.1, 0), S(200, 0)]), [S(200, 0)],
                 'only the LAST gap matters — everything before it is another episode')

// (a) hz spread the sample count across the dead gap.
const resumed = { jointsSeen: true, samples: [
  ...Array(10).fill(0).map((_, i) => S(1_000 + i * 0.1, 2)),      // 10 Hz, then the stream dies
  ...Array(10).fill(0).map((_, i) => S(1_600 + i * 0.1, 0.04)),   // ten minutes later, 10 Hz again
] }
const view = summarize(resumed, 1_600.9)
assert.ok(view.hz > 9 && view.hz < 11, `hz must describe the CURRENT episode, got ${view.hz.toFixed(2)} ` +
          '(before Q91 the dead gap dragged it to ~0.03 Hz while frames arrived at ten a second)')
assert.equal(view.samples.length, 10,
             'and the sparkline gets only the current episode — it plots by INDEX, so a ten-minute ' +
             'silence was drawn as one adjacent pixel, a line that looks like motion across an outage')

// (b) THE DANGEROUS ONE: a big move BEFORE the outage raised the bar for the move happening
// now. peak was the loudest motion anywhere in the ring and `moving` asks for >5% of it, so an
// arm creeping at 2% of its old peak was reported "still" — the one sentence on this card that
// gets a person's hands near the hardware.
assert.equal(view.moving, true,
             'Q91: motion of 0.04 is real motion when the current episode peaks at 0.04; judging it ' +
             'against a peak of 2 from before the silence returned "still - safe to approach"')

// (c) the age is reported even when the resume is too young to judge.
const justBack = summarize({ jointsSeen: true, samples: [...Array(9).fill(0).map((_, i) => S(1_000 + i * 0.1, 2)), S(1_600, 0.5)] }, 1_601)
assert.equal(justBack.moving, null, 'one sample into a new episode is no opinion, not an inherited one')
assert.ok(Math.abs(justBack.stateAgeS - 1) < 1e-6,
          'but the AGE still comes from the newest sample — it used to return null whenever the run was ' +
          'shorter than 2, which is exactly when statusSentence needs to say "the stream stopped Ns ago"')

console.log('telemetryRing.test.mjs: all assertions passed')
