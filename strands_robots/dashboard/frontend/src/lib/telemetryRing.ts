import type { Peer } from '../types'

/**
 * The state-topic telemetry ring, as pure functions.
 *
 * This logic used to live inside useTelemetry's body, which means it could only
 * be exercised by rendering a component — so the one judgment the robot card
 * makes about PHYSICAL MOTION ("the arm is moving", "measured stillness", "no
 * opinion") had no test at all, while statusSentence turns each of those into
 * an accusation: a wedged policy, or an arm moving with nobody commanding it.
 *
 * Keeping it here means run-lib-tests gates it, and the hook becomes a thin
 * wrapper that owns only React refs.
 */

export const TELEMETRY_CAP = 120 // ~12 s at the 10 Hz state topic

export interface TelemetrySample { t: number; motion: number }

export interface RingAcc {
  /** newest last, capped at TELEMETRY_CAP */
  samples: TelemetrySample[]
  /** joint values of the previous frame, for the per-frame delta */
  prev: number[]
  /** has ANY frame carried joint positions? null before the first frame */
  jointsSeen: boolean | null
  /** the state timestamp already folded in, to ignore repeated frames */
  lastT: number | undefined
}

export const emptyRing = (): RingAcc => ({ samples: [], prev: [], jointsSeen: null, lastT: undefined })

/**
 * Joint positions out of a state payload, whatever shape the robot published:
 * a bare number, a [position, velocity] pair, or an object with .position.
 */
export function jointValues(peer: Pick<Peer, 'state'>): number[] {
  const joints = peer.state?.joints
  if (!joints) return []
  return Object.values(joints).map(v => {
    if (typeof v === 'number') return v
    if (Array.isArray(v)) return v[0] ?? 0
    return (v as { position?: number }).position ?? 0
  })
}

/** Mean absolute change per joint between two frames. Unit-free by construction: whatever
 *  unit the joints are in (the real arms report DEGREES, the sim twin radians), motion is in
 *  that same unit — which is exactly why no absolute threshold may be applied to it. */
export function motionBetween(prev: number[], values: number[]): number {
  // A frame whose joint COUNT changed cannot be differenced against the previous one; report
  // no motion rather than a spike manufactured from a reshaped vector.
  if (prev.length !== values.length || values.length === 0) return 0
  let motion = 0
  for (let i = 0; i < values.length; i++) motion += Math.abs(values[i] - prev[i])
  return motion / values.length
}

/**
 * Fold one state frame into the ring. Pure: returns a new accumulator, or the SAME one when
 * the frame carries nothing new (undefined or already-seen timestamp), so a caller can use
 * identity to decide whether to re-render.
 */
export function advance(acc: RingAcc, peer: Pick<Peer, 'state'>, nowS: number): RingAcc {
  const stateT = peer.state?.t
  if (stateT === undefined || stateT === acc.lastT) return acc
  const values = jointValues(peer)
  return {
    samples: [...acc.samples, { t: nowS, motion: motionBetween(acc.prev, values) }].slice(-TELEMETRY_CAP),
    prev: values,
    // Once joints have been seen they stay seen: an arm that drops a frame has not stopped
    // being an arm, and `jointsSeen: false` is authoritative enough downstream to suppress
    // the whole motion sentence.
    jointsSeen: (acc.jointsSeen ?? false) || values.length > 0,
    lastT: stateT,
  }
}

/**
 * A gap in ARRIVALS longer than this starts a new episode. Same 5 s that
 * lib/statusSentence calls a stopped stream, deliberately: one number, so the strip cannot describe
 * a rate the sentence beside it calls stale.
 */
export const TELEMETRY_GAP_S = 5

/**
 * The trailing run of samples with no dead gap in it — the only ones that describe NOW.
 *
 * Samples are stamped by ARRIVAL and the ring is capped by COUNT, never by age, so a stream that
 * stops and resumes leaves one ring holding two episodes with a silence between them. Judging the
 * present from that mixture was wrong three different ways at once (Q91):
 *
 *  - `hz = (n-1)/span` spread the sample count across the dead gap: 10 frames at 10 Hz, ten minutes
 *    of silence, then 10 more at 10 Hz reads "0.03 Hz" while frames arrive at ten a second.
 *  - `peak` was the loudest motion anywhere in the ring, and `moving` asks whether recent motion
 *    exceeds 5% of it. A big move before the outage therefore RAISED THE BAR for the move happening
 *    now: an arm creeping at 2% of its old peak was reported "still — safe to approach", which is the
 *    one sentence on this card that gets a person's hands near the hardware.
 *  - the sparkline plots by INDEX, so a ten-minute silence was drawn as one adjacent pixel: a line
 *    that looks like continuous motion across an outage.
 *
 * On this fleet that mixture is routine, not exotic — the arms are respawned constantly, and both real
 * ones have spent days with a state topic that stops and starts.
 */
export function recentRun(samples: TelemetrySample[], maxGapS = TELEMETRY_GAP_S): TelemetrySample[] {
  for (let i = samples.length - 1; i > 0; i--) {
    if (samples[i].t - samples[i - 1].t > maxGapS) return samples.slice(i)
  }
  return samples
}

export interface TelemetryView {
  samples: TelemetrySample[]
  /** measured state-topic rate, Hz (0 until 2+ samples) */
  hz: number
  /** joints changed recently; null until enough samples to judge */
  moving: boolean | null
  /** the peer publishes joint positions at all; null before any state sample */
  jointsSeen: boolean | null
  /** seconds since the newest state sample; null before the first one */
  stateAgeS: number | null
}

/**
 * Derive the card's view of the ring.
 *
 * NO JOINTS MEANS NO OPINION. `motion` is computed from joint positions, so a peer that
 * publishes none yields motion 0 on every sample — and treating that as evidence returned
 * `false`, i.e. a MEASURED stillness manufactured out of an empty stream. That is how a card
 * reading "no joint data on this peer" came to also display "idle and still — safe to
 * approach" (live, so101-arm-1). Both of cagatay's REAL arms are in exactly this state today
 * (no `joints` key at all, two different root causes), so this is the common case, not the
 * exotic one.
 *
 * `moving` also stays null until 10 samples exist: the status sentence uses motion to accuse a
 * policy of being wedged or to warn about an uncommanded arm, and an accusation off 2 samples
 * is noise.
 */
export function summarize(acc: RingAcc, nowS: number): TelemetryView {
  const { jointsSeen } = acc
  // Only the CURRENT episode is evidence about now (Q91). See recentRun.
  const samples = recentRun(acc.samples)
  const newest = acc.samples[acc.samples.length - 1]
  if (samples.length < 2) {
    return { samples, hz: 0, moving: null, stateAgeS: newest ? nowS - newest.t : null, jointsSeen }
  }
  const span = samples[samples.length - 1].t - samples[0].t
  const hz = span > 0 ? (samples.length - 1) / span : 0
  const peak = Math.max(...samples.map(s => s.motion), 1e-6)
  const moving = jointsSeen && samples.length >= 10
    ? samples.slice(-10).some(s => s.motion > peak * 0.05)
    : null
  return { samples, hz, moving, stateAgeS: nowS - samples[samples.length - 1].t, jointsSeen }
}
