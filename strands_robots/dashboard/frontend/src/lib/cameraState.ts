/**
 * What a camera tile is actually showing (U13).
 *
 * A black tile is the least debuggable thing on this dashboard, and a FROZEN
 * tile is worse: a last-good frame sitting there at full brightness reads as
 * "live" to every human who looks at it, which is how someone ends up trusting
 * a picture of where the arm was ten seconds ago.
 *
 * So this is a pure classifier. Given what the socket has seen, it names one
 * state and phrases it as a fact ("last frame 4s ago"), never as a vibe, and
 * says whether the pixels underneath are still trustworthy.
 */
export type CamKind =
  | 'connecting' | 'live' | 'stalled' | 'silent' | 'waiting'
  | 'busy' | 'unauthorized' | 'error' | 'closed' | 'retrying'

export interface CamStatus {
  kind: CamKind
  /** short headline, e.g. "stalled" */
  title: string
  /** the fact behind it, e.g. "last frame 4s ago" */
  detail: string
  /** frames arrived recently: the image is worth showing at full strength */
  live: boolean
  /** an old frame is on screen and must be visibly marked as stale */
  frozen: boolean
}

/** No frame for this long and the picture is no longer the present. */
export const STALL_MS = 2500

/** macOS/Linux capture backends all have their own way of saying "taken". */
const BUSY = /(in use|busy|already open|cannot open|could not open|-11852|EBUSY|Resource temporarily unavailable)/i
const DENIED = /(unauthorized|not permitted|permission|denied|forbidden|TCC)/i

export function classifyCamera(input: {
  now: number
  conn: 'connecting' | 'open' | 'closed'
  frames: number
  lastFrameAt?: number
  error?: string | null
  /** the peer says it has published frames (from the presence snapshot) */
  publishedAt?: number
  /** when a reconnect is scheduled, and which attempt it will be */
  retryInMs?: number
  attempt?: number
  stallMs?: number
}): CamStatus {
  const { now, conn, frames, lastFrameAt, error, publishedAt } = input
  const stallMs = input.stallMs ?? STALL_MS
  const age = lastFrameAt === undefined ? undefined : now - lastFrameAt
  const hadFrames = frames > 0 && age !== undefined
  const secs = (ms: number) => (ms < 1000 ? '<1s' : `${Math.round(ms / 1000)}s`)

  if (error && DENIED.test(error)) {
    return { kind: 'unauthorized', title: 'not permitted', detail: 'this session may not read camera frames', live: false, frozen: hadFrames }
  }
  if (error && BUSY.test(error)) {
    return { kind: 'busy', title: 'camera busy', detail: 'another app is holding this device', live: false, frozen: hadFrames }
  }
  if (error) {
    return { kind: 'error', title: 'no image', detail: error, live: false, frozen: hadFrames }
  }
  // A stall outranks a healthy socket: the connection being fine is exactly
  // what makes a frozen frame convincing.
  if (hadFrames && age! > stallMs) {
    return { kind: 'stalled', title: 'stalled', detail: `last frame ${secs(age!)} ago`, live: false, frozen: true }
  }
  if (hadFrames) return { kind: 'live', title: 'live', detail: '', live: true, frozen: false }
  if (conn === 'closed') {
    if (input.retryInMs !== undefined) {
      return {
        kind: 'retrying', title: 'reconnecting',
        detail: `in ${secs(input.retryInMs)}${input.attempt ? ` (attempt ${input.attempt})` : ''}`,
        live: false, frozen: false,
      }
    }
    return { kind: 'closed', title: 'disconnected', detail: 'stream closed', live: false, frozen: false }
  }
  if (conn === 'open') {
    return publishedAt
      ? { kind: 'waiting', title: 'waiting', detail: 'peer published frames, none arrived yet', live: false, frozen: false }
      : { kind: 'silent', title: 'no frames', detail: 'stream open, camera sending nothing', live: false, frozen: false }
  }
  return { kind: 'connecting', title: 'connecting', detail: 'opening the stream', live: false, frozen: false }
}

/** Backoff for reconnects: quick twice, then slow enough to be polite. */
export function retryDelayMs(attempt: number): number {
  return Math.min(10_000, 1000 * Math.pow(2, Math.max(0, attempt - 1)))
}
