/**
 * Joint history: a bounded, time-ordered ring per joint (U6).
 *
 * A bar shows where a joint IS. It cannot show that the arm has been still for
 * a minute, or that it just slammed into a limit and came back - and those are
 * the things you actually want to see while a policy runs. So every state frame
 * is appended to a per-joint track and drawn left (older) to right (now).
 *
 * Time, not sample index, is the x axis. The state stream is not isochronous:
 * frames arrive at STATE_HZ when things are healthy and in clumps when they
 * are not, so plotting by index would quietly stretch a stall into smooth
 * motion. Plotting by timestamp makes a gap look like a gap, which is the
 * honest picture, and `gapAfter` marks the ones long enough that connecting
 * the two samples would be an invention.
 *
 * Everything here is pure (or explicitly documented as mutating the store the
 * caller owns) so the geometry can be reasoned about without a canvas.
 */
import type { Range } from './jointScale'

/** How much past the strips remember. 60s at 30Hz is ~1800 points per joint. */
export const HISTORY_WINDOW_MS = 60_000
/** Hard cap per track, so a fast stream cannot grow the buffer without bound. */
export const MAX_POINTS = 900
/** A hole this long is a hole, not a line. ~10 missed frames at 30Hz. */
export const GAP_MS = 400

export interface Sample {
  t: number
  v: number
}

export type History = Map<string, Sample[]>

export function createHistory(): History {
  return new Map()
}

/**
 * Append one state frame. MUTATES the passed history (it is a ref the caller
 * owns) and returns it, so a React component can keep it out of render state.
 * Joints that vanish from the stream keep their past until it ages out.
 */
export function pushFrame(
  history: History,
  samples: Array<[string, number]>,
  now: number,
  windowMs = HISTORY_WINDOW_MS,
): History {
  for (const [name, v] of samples) {
    if (!Number.isFinite(v)) continue
    let track = history.get(name)
    if (!track) {
      track = []
      history.set(name, track)
    }
    const last = track[track.length - 1]
    // The same frame can be rendered twice (StrictMode, a re-render with no new
    // state): an identical timestamp is the same instant, not new evidence.
    if (last && last.t === now) {
      last.v = v
      continue
    }
    track.push({ t: now, v })
    const cutoff = now - windowMs
    let drop = 0
    while (drop < track.length && track[drop].t < cutoff) drop++
    if (drop) track.splice(0, drop)
    if (track.length > MAX_POINTS) track.splice(0, track.length - MAX_POINTS)
  }
  return history
}

export interface TracePoint {
  x: number
  y: number
  /** true when the next point is far enough away that the line must break */
  gapAfter: boolean
}

/**
 * Project a track into canvas space. x maps the time window onto [0, w] with
 * "now" at the right edge; y maps the joint's own learned range onto [h, 0]
 * (inverted, because canvas y grows downward and up should mean "higher").
 * Values outside the range are clamped rather than dropped: a joint pinned at
 * its limit should draw a flat line on the edge, not disappear.
 */
export function traceFor(
  track: Sample[] | undefined,
  now: number,
  range: Range,
  w: number,
  h: number,
  windowMs = HISTORY_WINDOW_MS,
  gapMs = GAP_MS,
): TracePoint[] {
  if (!track || track.length === 0 || w <= 0 || h <= 0) return []
  const span = range.hi - range.lo
  const out: TracePoint[] = []
  const oldest = now - windowMs
  for (let i = 0; i < track.length; i++) {
    const s = track[i]
    if (s.t < oldest) continue
    const x = w - ((now - s.t) / windowMs) * w
    const frac = span > 0 ? (s.v - range.lo) / span : 0.5
    const y = h - Math.min(1, Math.max(0, frac)) * h
    const next = track[i + 1]
    out.push({ x, y, gapAfter: !!next && next.t - s.t > gapMs })
  }
  return out
}

/**
 * The label that tells the truth about how much history there is (Q156b).
 *
 * Both joint labels claimed "last 60 seconds of movement" from the first frame onward —
 * the sparkline of an arm that appeared three seconds ago was announced as a minute of
 * movement, and a flat trace then reads as a minute of STILLNESS rather than as a robot
 * that just arrived. For a screen-reader user that label is the entire chart, so the
 * claim was the whole picture and it was wrong.
 *
 * Narrows the CLAIM instead of faking the data, the same move connBadge makes for the
 * socket badge. heldSeconds() does the measuring and had no caller until now.
 */
export function historyClaim(
  subject: string,
  track: Sample[] | undefined,
  now: number,
  windowMs: number = HISTORY_WINDOW_MS,
): string {
  const windowS = Math.round(windowMs / 1000)
  const held = heldSeconds(track, now)
  // Under a second of span is not a window worth quoting: one frame, or two frames a
  // blink apart, is "nothing yet" to a reader deciding whether to trust a flat line.
  if (held < 1) return `no movement history for ${subject} yet`
  // A near-full window rounds to the honest round number: quibbling over the last 3%
  // (a frame that aged out between measure and paint) would make the label flicker.
  if (held >= windowS * 0.97) return `last ${windowS}s of ${subject}`
  return `${Math.round(held)}s of ${subject} so far — the ${windowS}s window is not full yet`
}

/** Seconds of history actually held, for the "60s" label to tell the truth. */
export function heldSeconds(track: Sample[] | undefined, now: number): number {
  if (!track || track.length < 2) return 0
  return Math.max(0, (now - track[0].t) / 1000)
}

/**
 * Has this track stopped receiving frames? Used by the sparkline to decide when
 * a redraw carries information: while data flows the parent's frame counter
 * drives the canvas, and only a stall needs the clock to keep sliding the window
 * (a reduced-motion reader must still see the gap open at the right edge).
 * An empty track is not stalled - it never started.
 */
export function stalled(track: Sample[] | undefined, now: number, gapMs = GAP_MS * 2): boolean {
  if (!track || track.length === 0) return false
  return now - track[track.length - 1].t > gapMs
}
