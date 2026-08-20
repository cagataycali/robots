/**
 * Which of a robot's cameras have STOPPED publishing — asked before the operator
 * commits an hour of their life to a recording.
 *
 * MEASURED on so101-arm-1: `top` publishing at 4fps, `wrist` last captured 10.4
 * HOURS earlier (its reader thread died: "exceeded maximum consecutive read
 * failures"). The camera tile is honest about that on its own. What was missing
 * was anything between a dead camera and the record button: the session takes the
 * follower's camera list from its profile, so the episodes would carry a frozen
 * image on every frame — discovered at TRAINING time, with the arm put away.
 *
 * The backend refuses this too (dashboard/camera_liveness.py, 409 +
 * `ignore_dead_cameras`). This exists so the operator learns it BEFORE the click
 * rather than from a rejected request: by the time they press record they have
 * usually already set up the scene.
 *
 * Same discipline as the server and as the arm-role warnings: POSITIVE EVIDENCE
 * ONLY. A camera with no capture time is not dead — nothing may have subscribed to
 * it yet — and a capture stamped in the future is clock skew between two machines,
 * which is not freshness and not death either. Silence never accuses.
 */

/** Captures older than this mean the camera stopped, not that it is between frames. */
export const CAMERA_STOPPED_AGE_S = 120

export interface CameraMeta { t?: number | null }

export interface StoppedCamera { camera: string; ageS: number }

/** Seconds since this camera's last capture, or null when unknowable. */
export function captureAge(meta: CameraMeta | undefined | null, nowS: number): number | null {
  const t = meta?.t
  if (t == null || typeof t !== 'number' || !Number.isFinite(t) || t <= 0) return null
  const age = nowS - t
  if (age < 0) return null // clock skew, not a very fresh frame
  return age
}

/**
 * The cameras we have evidence have stopped, oldest first (the worst offender
 * leads the sentence the operator reads).
 */
export function stoppedCameras(
  cameras: Record<string, CameraMeta> | undefined | null,
  nowS: number,
  maxAgeS: number = CAMERA_STOPPED_AGE_S,
): StoppedCamera[] {
  const out: StoppedCamera[] = []
  for (const [camera, meta] of Object.entries(cameras ?? {})) {
    const ageS = captureAge(meta, nowS)
    if (ageS !== null && ageS > maxAgeS) out.push({ camera, ageS })
  }
  return out.sort((a, b) => b.ageS - a.ageS)
}

/**
 * A duration with its own "ago" ALREADY IN IT — `12.3h ago`. Documented loudly because a
 * caller that appends one produces "12.3h ago ago", which is exactly what shipped on the
 * card note in fd01a7bd and what an operator saw on the live dashboard for half a day.
 */
export function agoText(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s ago`
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`
  return `${(seconds / 3600).toFixed(1)}h ago`
}

/**
 * The warning shown next to the follower picker, or null when there is nothing
 * to say. Names the camera, its age and the CONSEQUENCE — a dataset is the
 * expensive thing here, and "wrist is stale" does not tell an operator that the
 * episodes they are about to record will be unusable.
 */
export function cameraWarning(
  stopped: StoppedCamera[],
  opts: { peerId?: string } = {},
): string | null {
  if (stopped.length === 0) return null
  const which = stopped.map(c => `${c.camera} (last frame ${agoText(c.ageS)})`).join(', ')
  const who = opts.peerId ? `${opts.peerId}: ` : ''
  const plural = stopped.length > 1 ? 'cameras have' : 'camera has'
  return (
    `${who}${stopped.length} ${plural} stopped publishing — ${which}. ` +
    `Recording now writes episodes with a frozen or missing image stream, ` +
    `which you would only notice at training time.`
  )
}


/**
 * The card's own line about cameras that have stopped — a pure function so the wording is
 * testable, and so the ONE mistake this line has already made cannot come back: agoText()
 * carries its own "ago", and the first version of this note appended another one.
 *
 * Says the count against the total (1 of 2 reads differently from "a camera stopped"), then
 * names each dead camera with its age, worst first.
 */
export function deadCameraNote(stopped: StoppedCamera[], totalCameras: number): string | null {
  if (stopped.length === 0 || totalCameras === 0) return null
  const which = stopped.map(c => `${c.camera} ${agoText(c.ageS)}`).join(', ')
  const noun = totalCameras > 1 ? 'cameras' : 'camera'
  return `${stopped.length} of ${totalCameras} ${noun} stopped — ${which}`
}
