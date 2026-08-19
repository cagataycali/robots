/**
 * The record screen's contract with the backend (U8: teleop episode collection).
 *
 * There is no /api/record on the server yet - the exact endpoints this module
 * wants are specified in FRONTEND_HANDOFF.md at the repo root. Until they
 * exist, `getRecordApi()` returns a mock that behaves like the real thing
 * (episodes accumulate, stop yields a summary, discard removes) so the whole
 * screen is testable end-to-end today and flips to the live backend the moment
 * `GET /api/record/session` stops 404ing. The probe result is cached per page
 * load; a `mock: true` flag rides on the api object so the UI can say so
 * honestly instead of pretending a dataset was written.
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
  fps: number
}

export interface RecordApi {
  mock: boolean
  session(): Promise<RecordSession>
  /** open a session (creates/appends the dataset) */
  open(opts: { dataset: string; task: string; leader: string; follower: string; target_episodes: number }): Promise<RecordSession>
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
    async session() { return clone() },
    async open(opts) {
      s = { ...EMPTY, ...opts, episodes: [], phase: 'idle' }
      return clone()
    },
    async startEpisode() {
      if (!s.dataset) throw new Error('no open session')
      if (s.phase === 'recording') return clone()
      s.phase = 'recording'
      startedAt = Date.now()
      return clone()
    },
    async stopEpisode() {
      if (s.phase !== 'recording') return clone()
      const duration = (Date.now() - startedAt) / 1000
      s.episodes.push({
        index: s.episodes.length,
        frames: Math.max(1, Math.round(duration * s.fps)),
        duration_s: Math.round(duration * 10) / 10,
        thumbnails: {},
      })
      s.phase = 'idle'
      return clone()
    },
    async redoEpisode() {
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

/** Probe once per page load: real backend if /api/record/session answers. */
export function getRecordApi(): Promise<RecordApi> {
  if (!cached) {
    cached = api('/api/record/session')
      .then(() => makeReal())
      .catch(() => makeMock())
  }
  return cached
}
