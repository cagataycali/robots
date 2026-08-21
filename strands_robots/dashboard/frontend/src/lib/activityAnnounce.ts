/**
 * What a screen reader hears when a new line lands in the activity log (Q158).
 *
 * The audit sheet — the one surface that answers "who moved that arm" — had no role, no
 * name and no live region either, so it was as mute as the agent dock was before Q157.
 *
 * THE CONTRAST WITH THE DOCK IS THE DESIGN. Chat text arrives one token at a time, so a
 * live transcript stutters and must be silent (Q157). Activity entries arrive WHOLE and
 * seconds apart — announcing them is the reason someone opened this sheet. But two traps
 * sit in the naive version:
 *   * the sheet loads server history right after mounting, so a live list would read
 *     dozens of old entries aloud on open. Nothing before the sheet opened is news, hence
 *     `sinceT`.
 *   * an e-stop storm appends many lines at once. An atomic region speaks the NEWEST one
 *     and lets the list carry the rest, rather than queueing a paragraph of speech that
 *     outlives the emergency it describes.
 * A failure is named as a failure, because "stop → arm-1" spoken flatly sounds like it
 * worked.
 */
import { activityLine, type ActivityRow } from './activityLine'

/** activityLine's row type says nothing about WHEN or WHO — the two fields this rule needs —
 *  so they are required here structurally rather than by loosening that shared type. */
type TimedRow = ActivityRow & { t: number; source: string }

export function activityAnnouncement(latest: TimedRow | undefined | null, sinceT: number): string {
  if (!latest || !(latest.t > sinceT)) return ''
  const v = activityLine(latest)
  const lead = v.tone === 'bad' ? 'failed — ' : v.tone === 'warn' ? 'warning — ' : ''
  const where = v.target && v.target !== '—' ? ` on ${v.target}` : ''
  const note = v.note ? ` ${v.note}` : ''
  return `${lead}${latest.source} ${latest.action}${where}: ${v.title}${note}`.replace(/\s+/g, ' ').trim()
}
