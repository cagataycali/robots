/**
 * Where focus goes when an overlay opens, and whether it is given back — as pure rules.
 *
 * The choices used to be three chained `??`s inside useDialogFocus's requestAnimationFrame, reachable
 * only by rendering a component in a browser. Two of them were wrong in ways a11y bugs usually are:
 * silently, and only on some paths (Q92).
 */

export const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export interface FocusCandidate {
  /** the overlay marked this element with [data-autofocus] */
  autofocus?: boolean
  /** aria-label, then title */
  label?: string
  /** visible text */
  text?: string
}

/** A close affordance, by any of the three things a real close button actually carries. */
export function looksLikeClose(c: FocusCandidate): boolean {
  const words = `${c.label ?? ''} ${c.text ?? ''}`.trim().toLowerCase()
  if (/(^|\s)(close|dismiss)\b/.test(words)) return true
  // A bare glyph button is a close button on every sheet in this app, and it has no words at all.
  return /^[\u00d7\u2715\u2716\u274c\u2573x]$/.test((c.text ?? '').trim())
}

/**
 * Controls that MOVE HARDWARE or destroy work. Focus is not activation — but the first focusable in a
 * sheet can be `▶ run` or `⏺ start recording`, and a keyboard user who lands there and presses space
 * has commanded a real arm. So one of these is never chosen automatically; the dialog itself is.
 */
export function looksDangerous(c: FocusCandidate): boolean {
  const words = `${c.label ?? ''} ${c.text ?? ''}`.toLowerCase()
  return /\b(run|start|record|recording|stop|e-?stop|resume|delete|remove|despawn|deploy|train|calibrate|home|move|teleop|replay)\b/
    .test(words)
}

/**
 * Which candidate to focus, or 'container' for the dialog element itself.
 *
 * 'container' is a real answer, not a failure: focusing the dialog (tabindex="-1") puts a screen
 * reader in the sheet and makes Tab start inside it, which is the whole point. The old code instead
 * called `pick?.focus()` on undefined — a SILENT NO-OP that left focus on the nav chip behind the
 * sheet, i.e. exactly the Q58 bug it exists to fix, for every overlay whose content had not mounted
 * on the frame after it opened (each of these sheets fetches on open).
 */
export function focusPlan(candidates: FocusCandidate[]): number | 'container' {
  const autofocus = candidates.findIndex(c => c.autofocus)
  if (autofocus >= 0) return autofocus
  const close = candidates.findIndex(looksLikeClose)
  if (close >= 0) return close
  const safe = candidates.findIndex(c => !looksDangerous(c))
  if (safe >= 0) return safe
  return 'container'
}

/**
 * The element to hand focus back to on close, or null for "leave it alone".
 *
 * `document.activeElement` is `<body>` far more often than it looks: Safari does not focus a button
 * when you click it, which is most of cagatay's traffic (he opens this dashboard on an iPhone). Saving
 * body as the opener and focusing it on close does not restore anything — it drops focus to the top of
 * the document, so the next Tab starts from nowhere, which is one of the symptoms Q58 was filed about.
 */
export function rememberOpener(active: unknown, body?: unknown): { el: unknown } | null {
  if (!active || active === body) return null
  return { el: active }
}

/**
 * Whether to pull focus back to the opener. Only if the overlay still HAS focus (or focus fell to
 * nowhere when it unmounted) — if the operator has already clicked something else, yanking their
 * focus is worse than leaving it.
 */
export function shouldRestoreFocus(s: {
  activeInsideOverlay: boolean
  activeIsBody: boolean
  openerConnected: boolean
}): boolean {
  if (!s.openerConnected) return false
  return s.activeInsideOverlay || s.activeIsBody
}
