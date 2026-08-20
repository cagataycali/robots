/**
 * The record screen's contract with the backend (U8: teleop episode collection).
 *
 * The live implementation is /api/record (strands_robots/dashboard/record_api.py),
 * speaking exactly the shape below. `getRecordApi()` probes
 * `GET /api/record/session` once per page load and only a 404 - the route
 * genuinely missing, e.g. an older backend - selects the in-browser mock,
 * which behaves like the real thing (episodes accumulate, stop yields a
 * summary, discard removes) so the screen stays testable against any server.
 * A `mock: true` flag rides on the api object so the UI can say so honestly
 * instead of pretending a dataset was written.
 */
import { api, post } from './endpoints'

export interface EpisodeSummary {
  index: number
  frames: number
  duration_s: number
  /** dataURL or backend URL per camera, e.g. { top: '/api/record/thumb/0/top' } */
  thumbnails: Record<string, string>
  discarded?: boolean
}

export interface RecordSession {
  /** null when no session is open */
  dataset: string | null
  task: string
  /** which teleop pair records: leader drives follower */
  leader: string | null
  follower: string | null
  target_episodes: number
  episodes: EpisodeSummary[]
  /** 'idle' = between episodes, 'recording' = frames being captured */
  phase: 'idle' | 'recording'
  /** The rate written into the dataset - a DECLARATION, not a measurement. */
  fps: number
  /**
   * The rate frames were really captured at, or null before two frames exist.
   * LeRobot timestamps a frame positionally as `frame_index / fps`, so this
   * number is nowhere in the artifact: if the pair disagrees, only the session
   * knows (BUGS.md Q70).
   */
  fps_achieved?: number | null
  /**
   * Present ONLY when `fps_achieved` differs from `fps` by more than 10%. A
   * notice and never a block - the operator is holding a leader arm mid-session,
   * and the rate is not something they can change from that position.
   */
  fps_notice?: {
    declared_fps: number
    measured_fps: number
    /** how many times the timestamps are off, e.g. 5.36 */
    ratio: number
    /** true = captured slower than declared (timestamps squeezed together) */
    slower: boolean
    detail: string
  } | null
  /**
   * Present ONLY when a camera the session asked for never opened. The dataset
   * schema is built from the follower's first observation, so a missing camera
   * is silently absent from every episode - the operator has to hear about it
   * before they teleoperate ten of them, not after.
   */
  camera_notice?: {
    requested: string[]
    present: string[]
    missing: string[]
    message: string
  } | null
  /**
   * Present ONLY when the follower has held ONE POSE for the whole measuring
   * window (BUGS.md Q35). A Feetech bus answers position reads off the USB logic
   * rail while torque needs the 12V pack, so a tripped supply mid-episode records
   * at full fps with valid numbers and perfect counters - and the dataset is a
   * still life that teaches a policy to hold still. Absent means "nothing to say
   * OR not enough evidence yet": it is NOT a certificate that the arm is moving,
   * so it must never be rendered as reassurance.
   */
  motion_notice?: {
    still: true
    /** how long the pose has been held, seconds */
    seconds: number
    samples: number
    /** the largest peak-to-peak travel of any joint, in DEGREES */
    max_travel_deg: number
    quietest_joint: string
    message: string
  } | null
}

export interface RecordApi {
  mock: boolean
  session(): Promise<RecordSession>
  /** open a session (creates/appends the dataset) */
  /**
   * `ignore_dead_cameras` is the operator's deliberate override of the server's
   * refusal when a configured camera has stopped publishing (Q45). Optional and
   * never defaulted: its absence means "let the server decide".
   */
  open(opts: { dataset: string; task: string; leader: string; follower: string; target_episodes: number; ignore_dead_cameras?: boolean }): Promise<RecordSession>
  startEpisode(): Promise<RecordSession>
  /** stop and keep the in-flight episode */
  stopEpisode(): Promise<RecordSession>
  /** stop and immediately discard the in-flight episode (redo) */
  redoEpisode(): Promise<RecordSession>
  discard(index: number): Promise<RecordSession>
  /** close the session; upload=true pushes the dataset to the HF Hub */
  close(opts?: { upload?: boolean; repo_id?: string }): Promise<{ ok: boolean; detail?: string }>
}

const EMPTY: RecordSession = {
  dataset: null, task: '', leader: null, follower: null,
  target_episodes: 10, episodes: [], phase: 'idle', fps: 30,
}

/* ------------------------------ mock ------------------------------ */

function makeMock(): RecordApi {
  let s: RecordSession = { ...EMPTY }
  let startedAt = 0
  const clone = () => JSON.parse(JSON.stringify(s)) as RecordSession
  return {
    mock: true,
    async session() {
      // tick the in-flight take's frames, as the real control loop would
      if (s.phase === 'recording' && s.episodes.length > 0) {
        const ep = s.episodes[s.episodes.length - 1]
        ep.frames = Math.max(0, Math.round(((Date.now() - startedAt) / 1000) * s.fps))
        ep.duration_s = Math.round(((Date.now() - startedAt) / 1000) * 10) / 10
      }
      return clone()
    },
    async open(opts) {
      s = { ...EMPTY, ...opts, episodes: [], phase: 'idle' }
      return clone()
    },
    async startEpisode() {
      if (!s.dataset) throw new Error('no open session')
      if (s.phase === 'recording') return clone()
      s.phase = 'recording'
      startedAt = Date.now()
      // The real backend lists the take in flight as the last episode entry,
      // frames growing as they are captured - mirror that so the UI's live
      // frame tick behaves identically against both.
      s.episodes.push({ index: s.episodes.length, frames: 0, duration_s: 0, thumbnails: {} })
      return clone()
    },
    async stopEpisode() {
      if (s.phase !== 'recording') return clone()
      const duration = (Date.now() - startedAt) / 1000
      const ep = s.episodes[s.episodes.length - 1]
      ep.frames = Math.max(1, Math.round(duration * s.fps))
      ep.duration_s = Math.round(duration * 10) / 10
      s.phase = 'idle'
      return clone()
    },
    async redoEpisode() {
      if (s.phase === 'recording') s.episodes.pop()
      s.phase = 'idle'
      return clone()
    },
    async discard(index) {
      const ep = s.episodes.find(e => e.index === index)
      if (ep) ep.discarded = true
      return clone()
    },
    async close() {
      s = { ...EMPTY }
      return { ok: true, detail: 'mock session closed (nothing was written)' }
    },
  }
}

/* ------------------------------ real ------------------------------ */

function makeReal(): RecordApi {
  return {
    mock: false,
    session: () => api<RecordSession>('/api/record/session'),
    open: opts => post<RecordSession>('/api/record/open', opts),
    startEpisode: () => post<RecordSession>('/api/record/episode/start'),
    stopEpisode: () => post<RecordSession>('/api/record/episode/stop'),
    redoEpisode: () => post<RecordSession>('/api/record/episode/redo'),
    discard: index => post<RecordSession>('/api/record/episode/discard', { index }),
    close: opts => post('/api/record/close', opts ?? {}),
  }
}

let cached: Promise<RecordApi> | null = null

/** Probe once per page load: real backend unless the route truly does not
 * exist. Only a 404 selects the in-browser rehearsal - a 401 means the
 * backend IS there and the auth gate will sort the token out, and a network
 * blip must not silently swap a real recorder for a mock that pretends to
 * write datasets. */
export function getRecordApi(): Promise<RecordApi> {
  if (!cached) {
    cached = api('/api/record/session')
      .then(() => makeReal())
      .catch(e => (e && (e as { status?: number }).status === 404 ? makeMock() : makeReal()))
  }
  return cached
}
