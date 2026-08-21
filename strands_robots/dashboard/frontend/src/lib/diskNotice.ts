/**
 * How the record screen says "the disk is running out" (Q92).
 *
 * The backend's `disk_notice` (dashboard/disk_headroom.py) is level-based and situation-blind: it
 * knows how much space is left, not what the operator is in the middle of. That difference decides
 * the ADVICE, which is the only part of a warning anybody acts on:
 *
 *  - BEFORE a session, "free space first" is actionable — nothing is running, and the operator can
 *    go delete a checkpoint and come back.
 *  - DURING a session it is not. Freeing space means leaving two parked arms and a half-written
 *    dataset to go hunting for files, and the honest instruction is the opposite: STOP at the end of
 *    the current episode, because episodes already written are complete and safe, and the damage
 *    only happens to the episode that runs out mid-write.
 *
 * So a critical disk mid-recording gets different words than the same disk on the idle form, and
 * neither of them ever blocks the operator: they are holding a leader arm.
 */

export type DiskLevel = 'tight' | 'critical'

export interface DiskNotice {
  level?: DiskLevel | string | null
  free_mb?: number | null
  headline?: string | null
  advice?: string | null
}

export interface DiskNoticeView {
  /** css tone class suffix used by train-msg */
  tone: 'warn' | 'bad'
  /** true only when this deserves role="alert" — a tight disk repeated at 1Hz must not shout */
  urgent: boolean
  headline: string
  advice: string
  /** stable hook for the audits */
  testid: string
}

/**
 * Returns null when there is nothing to render — no notice, an unknown level, or a notice with no
 * headline. A malformed payload must render NOTHING rather than an empty warning box: the record
 * screen already carries three notices, and a fourth that appears blank teaches the operator that
 * warnings here mean nothing.
 */
export function diskNoticeView(
  notice: DiskNotice | null | undefined,
  opts: { recording?: boolean } = {},
): DiskNoticeView | null {
  if (!notice) return null
  const level = notice.level === 'critical' ? 'critical' : notice.level === 'tight' ? 'tight' : null
  if (!level) return null
  const headline = (notice.headline ?? '').trim()
  if (!headline) return null
  const backendAdvice = (notice.advice ?? '').trim()
  const recording = !!opts.recording

  if (level === 'critical' && recording) {
    return {
      tone: 'bad',
      urgent: true,
      headline,
      // Deliberately NOT the backend's "free space first": that is unreachable advice for someone
      // holding an arm over a live dataset.
      advice:
        'Stop after this episode. The episodes already written are complete and safe — it is the ' +
        'one that runs out mid-write that leaves a dataset whose meta promises more than its data ' +
        'holds. Close the session, free space, then open a new one and keep going.',
      testid: 'disk-critical-recording',
    }
  }
  if (level === 'critical') {
    return { tone: 'bad', urgent: true, headline, advice: backendAdvice, testid: 'disk-critical' }
  }
  return {
    tone: 'warn',
    // A tight disk is a fact to notice, not an emergency, and this document is polled about once a
    // second — role="alert" on it would interrupt a screen reader repeatedly for unchanged news.
    urgent: false,
    headline,
    advice: backendAdvice,
    testid: recording ? 'disk-tight-recording' : 'disk-tight',
  }
}
