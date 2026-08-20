/**
 * Q54: the rate a recording session DECLARES — and the one honest thing to do with the measured
 * rate when it disagrees.
 *
 * /api/record/open has always accepted `fps` (record_api.py: `fps=int(body.get("fps", 30) or 30)`)
 * and the record form never sent it, so every dataset this dashboard has ever produced is stamped
 * 30 fps. On these SO-101s that is not close: each frame is one bus-locked serial read, so capture
 * lands around 4 Hz — and LeRobot timestamps a frame positionally as frame_index / fps, which
 * means a dataset recorded at 4 Hz and declared at 30 says every motion happened 7x faster than
 * it did. A policy trained on it learns the wrong speed, and nothing in the artifact records the
 * discrepancy. The panel already SHOWED the gap (fps_notice) but the number it was complaining
 * about could not be changed from any screen: a warning with no remedy.
 *
 * Pure module. Two jobs: validate what the operator typed, and turn a measured rate into the
 * value they would want next time — never applied silently, because re-declaring the rate is a
 * decision about the dataset's meaning.
 */

export interface FpsField {
  /** the number to send; only meaningful when `problem` is null */
  value: number
  /** why this input cannot be used, in the operator's words — null when fine */
  problem: string | null
  /** a change we DID make and therefore must admit (30.6 → 30) */
  note: string | null
}

export const DEFAULT_FPS = 30
const MIN_FPS = 1
const MAX_FPS = 60

/** Parse the fps box. Mirrors episodeTarget's posture: correct nothing in silence. */
export function fpsField(raw: string): FpsField {
  const text = (raw ?? '').trim()
  if (!text) return { value: DEFAULT_FPS, problem: null, note: null }
  const n = Number(text)
  if (!Number.isFinite(n)) {
    return { value: DEFAULT_FPS, problem: `"${text}" is not a number of frames per second`, note: null }
  }
  if (n < MIN_FPS || n > MAX_FPS) {
    return {
      value: DEFAULT_FPS,
      problem: `fps must be between ${MIN_FPS} and ${MAX_FPS} — ${n} is outside what a dataset can declare`,
      note: null,
    }
  }
  const rounded = Math.round(n)
  return {
    value: rounded,
    problem: null,
    note: rounded !== n ? `recording at ${rounded} fps (a dataset's rate is a whole number)` : null,
  }
}

export interface FpsNoticeLike {
  declared_fps: number
  measured_fps: number
  ratio: number
  slower: boolean
  detail: string
}

export interface FpsSuggestion {
  /** the value to put in the box, as text (the field is raw text) */
  fps: string
  /** the button's words */
  label: string
  /** why, including what the mismatch costs — shown next to the button */
  why: string
}

/**
 * What to offer when the measured rate disagrees with the declared one.
 *
 * Returns null when there is nothing honest to suggest: no notice, or a measurement too small to
 * round into a legal rate (a session that captured 0.4 frames a second has a bigger problem than
 * its declaration, and suggesting "1 fps" would dress that up as a fix).
 */
export function fpsSuggestion(notice: FpsNoticeLike | null | undefined): FpsSuggestion | null {
  if (!notice || !Number.isFinite(notice.measured_fps)) return null
  const measured = Math.round(notice.measured_fps)
  if (measured < MIN_FPS || measured > MAX_FPS) return null
  if (measured === Math.round(notice.declared_fps)) return null
  return {
    fps: String(measured),
    label: `use ${measured} fps next session`,
    // Deliberately future-tense: the current session's episodes keep their declaration, and
    // pretending otherwise would be the third lie in this chain.
    why: notice.slower
      ? `this session is already stamped ${notice.declared_fps} fps and cannot be re-declared; `
        + `recording the next one at ${measured} makes its timestamps match real time`
      : `capture is running faster than declared; ${measured} fps would describe it honestly`,
  }
}
