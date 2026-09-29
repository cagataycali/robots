/**
 * Colour scheme, with the docs' semantics: the OS preference is the default,
 * and a choice the operator makes on this device is remembered and wins.
 *
 * The two schemes are the docs' `paper` (white page, black ink) and `dark`
 * (black page, white ink); `styles.css` keys every token on
 * `html[data-scheme]`. No attribute means "follow the OS", which is why the
 * stored value is removed, not set, when the operator picks "auto" again.
 */
export type Scheme = 'paper' | 'dark'

export const SCHEME_KEY = 'strands-dash-scheme'

/** The stored choice, or null when the page follows the OS. */
export function storedScheme(storage: Pick<Storage, 'getItem'> = localStorage): Scheme | null {
  const v = storage.getItem(SCHEME_KEY)
  return v === 'paper' || v === 'dark' ? v : null
}

/** What the OS asks for right now. */
export function systemScheme(matches: boolean = window.matchMedia('(prefers-color-scheme: dark)').matches): Scheme {
  return matches ? 'dark' : 'paper'
}

/** The scheme in effect: the stored choice, else the OS. */
export function effectiveScheme(stored: Scheme | null, system: Scheme): Scheme {
  return stored ?? system
}

/** The scheme a toggle press moves to: the opposite of what is on screen. */
export function nextScheme(current: Scheme): Scheme {
  return current === 'dark' ? 'paper' : 'dark'
}

/** Paint a scheme on the document (null = follow the OS). */
export function applyScheme(scheme: Scheme | null, root: HTMLElement = document.documentElement): void {
  if (scheme) root.dataset.scheme = scheme
  else delete root.dataset.scheme
}

/** Remember a choice on this device and paint it. */
export function chooseScheme(scheme: Scheme, storage: Pick<Storage, 'setItem'> = localStorage): void {
  storage.setItem(SCHEME_KEY, scheme)
  applyScheme(scheme)
}

/** Restore the stored choice before the first paint. Safe to call twice. */
export function initScheme(): void {
  try { applyScheme(storedScheme()) } catch { /* storage denied: follow the OS */ }
}
