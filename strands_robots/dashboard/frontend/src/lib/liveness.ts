/**
 * What counts as PROOF that a peer is still alive.
 *
 * `last_seen` is stamped in the browser whenever a mesh event arrives, and the
 * stale sweep turns it into "no heartbeat for 20s - treat the arm as
 * unpredictable". So every event type that refreshes it is making a claim about
 * a process on the other side of the mesh.
 *
 * A CAMERA frame is not such a proof. The dashboard replays a camera's last
 * cached frame to each new subscriber (measured: a frame captured 6.8 hours
 * earlier arrived the instant a tile mounted, see cameraState.ts), so opening a
 * page can deliver camera events for a peer that died hours ago - and that
 * refreshed last_seen, cleared `stale`, and made a dead robot's card read
 * "seen 0s ago" with a live green dot.
 *
 * The frame carries the CAPTURE time the peer put on it, so the question is
 * answerable rather than a guess: recent capture = the peer really is publishing,
 * old capture = a replay from our own cache and no evidence of anything.
 */

/** A capture older than this proves nothing about liveness (the stale window). */
export const FRAME_LIVENESS_MAX_AGE_S = 15

export function frameProvesLiveness(input: {
  /** capture time the peer stamped on the frame, epoch seconds */
  frameT?: number | null
  /** browser now, epoch seconds */
  nowS: number
  maxAgeS?: number
}): boolean {
  const { frameT, nowS } = input
  const maxAgeS = input.maxAgeS ?? FRAME_LIVENESS_MAX_AGE_S
  if (frameT == null || !Number.isFinite(frameT) || frameT <= 0) {
    // No capture time at all: unknowable, so it does not get to vouch for the
    // peer. Presence and state events still refresh liveness on their own.
    return false
  }
  const age = nowS - frameT
  // A capture stamped in the FUTURE is clock skew between two machines, not
  // freshness - but it is also not evidence of death, so it is simply not
  // counted either way (and never allowed to look newer than now).
  if (age < 0) return false
  return age <= maxAgeS
}
