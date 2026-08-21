import { useEffect, useRef, type RefObject } from 'react'
import { FOCUSABLE, focusPlan, rememberOpener, shouldRestoreFocus } from './dialogFocus'

/**
 * When an overlay opens, focus goes INSIDE it — and comes back to whatever opened it on close.
 *
 * Measured before this existed (BUGS.md Q58): opening devices, record, train, activity, settings
 * or a robot's detail screen left focus on the nav chip BEHIND the sheet. So Tab walked the fleet
 * the sheet was covering, Escape-then-Tab started from nowhere in particular, and a screen reader
 * was still reading the page the user had just left. HelpSheet was the only overlay that moved
 * focus (`closeRef.current?.focus()`), and even it never gave focus back.
 *
 * What it focuses, in order of preference:
 *   1. `[data-autofocus]` — the overlay says where it wants to start
 *   2. its close button — like HelpSheet, and the safe default: focus is not activation, but the
 *      first focusable in a sheet can be `▶ run` or `⏺ start recording`, and a keyboard user who
 *      lands there and presses space has commanded a real arm
 *   3. the first focusable element
 *
 * This is NOT a focus trap. Trapping is a bigger change with its own failure mode (a trap with a
 * bug locks the operator inside a sheet while an arm is moving); getting focus in and out
 * correctly is the part that was actually broken.
 */
export function useDialogFocus(ref: RefObject<HTMLElement | null>, open = true): void {
  const opener = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (!open) return
    // Safari does not focus a button when it is clicked, so the "opener" is often <body> - and giving
    // focus BACK to body drops it to the top of the document (Q92). ./dialogFocus decides.
    const remembered = rememberOpener(document.activeElement, document.body)
    opener.current = remembered?.el instanceof HTMLElement ? remembered.el : null
    // One frame later: the overlay's children may still be mounting on the open tick.
    const id = requestAnimationFrame(() => {
      const node = ref.current
      if (!node || node.contains(document.activeElement)) return
      const els = [...node.querySelectorAll<HTMLElement>(FOCUSABLE)]
      const plan = focusPlan(els.map(el => ({
        autofocus: el.hasAttribute('data-autofocus'),
        label: el.getAttribute('aria-label') ?? el.getAttribute('title') ?? '',
        text: el.textContent ?? '',
      })))
      if (plan === 'container') {
        // A real answer, not a failure: focusing the dialog puts a screen reader inside it and makes
        // Tab start inside it. The old code called focus() on undefined here - a silent no-op that
        // left focus on the nav chip BEHIND the sheet whenever nothing focusable had mounted yet.
        if (!node.hasAttribute('tabindex')) node.setAttribute('tabindex', '-1')
        node.focus()
        return
      }
      els[plan]?.focus()
    })
    return () => {
      cancelAnimationFrame(id)
      const active = document.activeElement
      if (shouldRestoreFocus({
        activeInsideOverlay: ref.current?.contains(active) ?? false,
        activeIsBody: active === document.body,
        openerConnected: opener.current?.isConnected ?? false,
      })) opener.current?.focus()
    }
  }, [ref, open])
}
