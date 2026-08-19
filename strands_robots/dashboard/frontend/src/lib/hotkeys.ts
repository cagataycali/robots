/**
 * The dashboard's global keys — JOURNEYS #12 ("11 tab stops to the e-stop on
 * the armed-run screen").
 *
 * Re-measured on the live page, and it was worse than the audit said: **14 to
 * 30+ tab stops** to reach STOP ALL depending on the screen, and on the
 * training screen it was **not reachable inside 30 tabs at all**. Two separate
 * causes, so two separate fixes:
 *
 * 1. The button was late in the DOM. It now renders FIRST in `App.tsx`, so
 *    Tab from the top of the document reaches it in one stop (it is
 *    `position: fixed` in its own layer — nothing moved on screen).
 * 2. Tab only goes FORWARD from where you last clicked, so DOM order cannot
 *    help someone already deep in a drawer — and the bare `.` hotkey is
 *    deliberately swallowed while typing, otherwise every task sentence
 *    containing a full stop would fire the stop sheet. That left the worst
 *    case exactly where the audit found it: an operator mid-sentence in the
 *    record or run form, many tabs from the brake.
 *
 * So `Cmd+.` / `Ctrl+.` fires the stop sheet **always, including inside a text
 * field**. A modified key cannot be part of the text being typed, which is the
 * whole reason it is safe where the bare key is not.
 */

export type HotkeyAction = 'estop' | 'help' | 'close' | null

/** The parts of a KeyboardEvent this decision actually depends on. */
export interface KeyLike {
  key: string
  metaKey?: boolean
  ctrlKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
  /** Tag name of the event target, e.g. 'INPUT' — case-insensitive. */
  targetTag?: string
  /** True when the target is contenteditable (a rich text surface). */
  editable?: boolean
  repeat?: boolean
}

const TYPING_TAGS = /^(INPUT|TEXTAREA|SELECT)$/i

export function isTyping(e: KeyLike): boolean {
  return !!e.editable || TYPING_TAGS.test(e.targetTag ?? '')
}

export function hotkeyVerdict(e: KeyLike): HotkeyAction {
  // Escape belongs to whatever is on top, everywhere, always — including while
  // typing, because "I want out of this form" is the same intent.
  if (e.key === 'Escape') return 'close'

  // THE BRAKE, reachable from inside a half-typed task sentence. Alt is left
  // out on purpose: Alt+. types characters on several keyboard layouts.
  const stopChord = (e.metaKey || e.ctrlKey) && !e.altKey && e.key === '.'
  if (stopChord) return 'estop'

  // Bare keys only outside text entry: a task sentence ends in a full stop, and
  // "?" is a character people type.
  if (isTyping(e)) return null
  if (e.altKey || e.metaKey || e.ctrlKey) return null
  if (e.key === '.') return 'estop'
  if (e.key === '?') return 'help'
  return null
}

/** What to advertise on the button (`aria-keyshortcuts` syntax). */
export const ESTOP_KEYSHORTCUTS = '. Meta+. Control+.'
