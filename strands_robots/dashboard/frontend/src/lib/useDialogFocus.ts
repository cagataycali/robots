import { useEffect, useRef, type RefObject } from 'react'

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
const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export function useDialogFocus(ref: RefObject<HTMLElement | null>, open = true): void {
  const opener = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement
    opener.current = previous instanceof HTMLElement ? previous : null
    // One frame later: the overlay's children may still be mounting on the open tick.
    const id = requestAnimationFrame(() => {
      const node = ref.current
      if (!node || node.contains(document.activeElement)) return
      const pick = node.querySelector<HTMLElement>('[data-autofocus]')
        ?? [...node.querySelectorAll<HTMLElement>('button')].find(b => /^close/i.test(b.getAttribute('aria-label') ?? ''))
        ?? node.querySelector<HTMLElement>(FOCUSABLE)
      pick?.focus()
    })
    return () => {
      cancelAnimationFrame(id)
      // Only take focus back if the overlay still holds it — if the user has already clicked
      // elsewhere, yanking their focus is worse than leaving it.
      const active = document.activeElement
      const inside = ref.current?.contains(active) ?? false
      if ((inside || active === document.body) && opener.current?.isConnected) opener.current.focus()
    }
  }, [ref, open])
}
