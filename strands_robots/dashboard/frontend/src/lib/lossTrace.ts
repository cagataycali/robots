/**
 * Loss history for a training job, accumulated from polled status snapshots.
 *
 * The trainer's status endpoint reports only the LATEST (step, loss) pair -
 * it parses the tail of the train log, there is no history API. So the
 * dashboard builds the curve itself from what it witnesses: each poll may
 * contribute one point. That shapes the rules here:
 *
 * - the same step reported twice is ONE point (polls outpace log lines)
 * - a step LOWER than the last seen means the job restarted; the old curve
 *   belongs to a run that no longer exists, so the trace resets rather than
 *   drawing a lying sawtooth
 * - non-finite loss (NaN in the log) is dropped from the trace but the
 *   caller can still surface it - a NaN loss is diagnostic gold, just not
 *   plottable
 * - capped: beyond `cap` points every second point of the OLDER half is
 *   dropped, preserving recent detail while keeping the early shape
 */

export interface LossPoint {
  step: number
  loss: number
}

export function pushLoss(
  trace: LossPoint[],
  step: unknown,
  loss: unknown,
  cap = 240,
): LossPoint[] {
  const s = typeof step === 'number' && Number.isFinite(step) ? step : null
  const l = typeof loss === 'number' && Number.isFinite(loss) ? loss : null
  if (s === null || l === null) return trace

  const last = trace[trace.length - 1]
  if (last) {
    if (s === last.step) {
      // same step, possibly a fresher loss reading - replace, don't append
      if (l === last.loss) return trace
      return [...trace.slice(0, -1), { step: s, loss: l }]
    }
    if (s < last.step) return [{ step: s, loss: l }] // restart detected
  }

  let next = [...trace, { step: s, loss: l }]
  if (next.length > cap) {
    const half = Math.floor(next.length / 2)
    next = [...next.slice(0, half).filter((_, i) => i % 2 === 0), ...next.slice(half)]
  }
  return next
}

/**
 * The vertical band a trace is drawn in — and whether it had to be INVENTED.
 *
 * A sparkline that always scales to min..max is a liar about flat data: a run stuck at loss
 * 2.5000 ± 0.0004 fills the full height with a dramatic mountain range, and the operator reads
 * progress that is not happening. Amplifying noise to 34 pixels is not neutral drawing; it is a
 * claim about the run.
 *
 * So the band has a FLOOR of 2% of the values' own magnitude. Real variation (a loss falling from
 * 2.5 to 0.8) is far wider than the floor and scales exactly as before; negligible variation gets
 * a band it cannot fill, so a flat run looks flat and a padded band is reported as `flat: true`
 * for the caller to say so in words.
 */
export function lossBand(points: readonly LossPoint[]): { lo: number; hi: number; flat: boolean } {
  let lo = Infinity
  let hi = -Infinity
  for (const p of points) {
    if (p.loss < lo) lo = p.loss
    if (p.loss > hi) hi = p.loss
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1, flat: true }
  const spread = hi - lo
  const magnitude = Math.max(Math.abs(hi), Math.abs(lo))
  // All-zero losses have no magnitude to take a percentage of, so the floor is absolute there.
  const floorSpan = magnitude > 0 ? magnitude * 0.02 : 1
  if (spread >= floorSpan) return { lo, hi, flat: false }
  const mid = (lo + hi) / 2
  // Centred, so a perfectly flat curve draws through the MIDDLE — pinned to the bottom edge (what
  // the old `Math.max(1e-9, hi - lo)` produced) reads as "converged to its best value".
  return { lo: mid - floorSpan / 2, hi: mid + floorSpan / 2, flat: true }
}

/** Scale points into canvas space. Returns [] for <2 points (nothing drawable). */
export function lossPath(
  points: LossPoint[],
  width: number,
  height: number,
  pad = 2,
): Array<[number, number]> {
  if (points.length < 2 || width <= 0 || height <= 0) return []
  const s0 = points[0].step
  const s1 = points[points.length - 1].step
  const { lo, hi } = lossBand(points)
  const sSpan = Math.max(1e-9, s1 - s0)
  const lSpan = Math.max(1e-9, hi - lo)
  const w = width - pad * 2
  const h = height - pad * 2
  return points.map(p => [
    pad + ((p.step - s0) / sSpan) * w,
    // loss falls downward on the chart: low loss = low y? No - low loss is
    // GOOD, so it sits at the BOTTOM (large y), matching every loss curve
    // a practitioner has ever seen.
    pad + (1 - (p.loss - lo) / lSpan) * h,
  ])
}

/** "12.3k" style step label - matches the trainer's own log formatting. */
export function fmtStep(step: number): string {
  if (!Number.isFinite(step)) return '?'
  if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`
  if (step >= 1_000) return `${(step / 1_000).toFixed(1)}k`
  return String(step)
}
