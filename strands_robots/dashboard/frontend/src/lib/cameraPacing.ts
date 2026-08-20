/**
 * When the SERVER says it is pacing this tile, what should the tile do about it?
 *
 * Q53 added a server-side churn guard because our client-side cure only ever reached
 * clients that reload: one tab reopened a camera 1.53 times a second for twelve hours,
 * hours after the fix meant for it had landed. The server now caps such a viewer and sends
 * a `camera_error` frame saying so — deliberately that frame type, because an old bundle
 * renders it.
 *
 * But for THIS bundle, routing it into `error` would be a lie: the stream is working, it is
 * merely slower than it was asked to be. A red "camera error" on a tile that is delivering
 * pictures is the same dishonesty as a green card on a locked-out arm (Q43) — the operator
 * reads a broken camera and goes hunting for a USB cable that is perfectly fine.
 *
 * So a throttle notice becomes three things instead of an error:
 *   1. a calm note the tile shows,
 *   2. the rate WE now ask for, so the next socket stops requesting a firehose the server
 *      has already refused (agreement costs nothing and halves the argument),
 *   3. nothing else — it is not an error, not a stall, and not a reason to stop retrying.
 */

export interface PacingNotice {
  /** the frame rate the server said it is serving us at, when it named one */
  fps: number | null
  /** what the tile shows, in the operator's words */
  note: string
}

/** Extract the number out of the server's sentence ("...pacing it at 2 fps until...") */
export function pacedFps(message: string): number | null {
  const m = /at\s+([0-9]+(?:\.[0-9]+)?)\s*fps/i.exec(message)
  if (!m) return null
  const v = Number(m[1])
  return Number.isFinite(v) && v > 0 ? v : null
}

/**
 * A `camera_error` payload -> a pacing notice, or null when it is a REAL error.
 *
 * The discriminator is the explicit `throttled` flag, never the wording: a server that
 * changes its sentence must not silently turn a paced tile back into a broken one.
 */
export function pacingFromNotice(ev: unknown): PacingNotice | null {
  if (!ev || typeof ev !== 'object') return null
  const e = ev as { type?: unknown; throttled?: unknown; error?: unknown }
  if (e.type !== 'camera_error' || e.throttled !== true) return null
  const message = typeof e.error === 'string' ? e.error : ''
  const fps = pacedFps(message)
  return {
    fps,
    note: fps === null
      ? 'the server is pacing this stream while it settles'
      : `paced by the server at ${fps} fps — the link was being saturated`,
  }
}

/**
 * The rate this tile should ask for next, given what it knows about itself and what the
 * server has told it. The LOWER wins: a tile that already decided to degrade itself does
 * not get talked back up by a server cap, and a server cap is not an invitation to ask
 * for more than we were going to.
 */
export function nextRequestedFps(own: number | null, paced: number | null): number | null {
  const caps = [own, paced].filter((v): v is number => typeof v === 'number' && v > 0)
  return caps.length ? Math.min(...caps) : null
}
