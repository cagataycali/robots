import { routeKnown } from './serverAge'
import { BUNDLE_ROUTES } from './bundleRoutes.generated'

/**
 * Which features on this page are DARK — routes the bundle calls that the running server does not have?
 *
 * lib/serverAge already explains one 404 after the fact. That is the right sentence at the wrong
 * moment: on this very machine the running dashboard lacked TEN routes the UI calls for three days,
 * so ten features answered a click with a refusal and nothing said so beforehand. The operator only
 * ever needed one fact — this server is older than this page, restart it — and they were made to
 * rediscover it per feature.
 *
 * SILENCE WHEN WE DO NOT KNOW: with no openapi.json (not fetched, unreachable, or a server that does
 * not publish one) this returns [] — never "everything is dark". routeKnown returns null for exactly
 * that case, and rendering our own missing fetch as an accusation about the server is the failure
 * serverAge's own comment warns about.
 */
export function darkRoutes(livePaths: string[] | null | undefined, needed: readonly string[] = BUNDLE_ROUTES): string[] {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return []
  return needed.filter(p => routeKnown(livePaths, p) === false).sort()
}

/** The banner sentence, or null when there is nothing to say. Names the count, the remedy, and why the
 *  remedy must be a terminal (a launchd-descended process can never be granted camera access on macOS). */
export function darkFeatureMessage(dark: readonly string[]): string | null {
  if (dark.length === 0) return null
  const n = dark.length
  return (
    `${n} ${n === 1 ? 'feature' : 'features'} on this page ${n === 1 ? 'is' : 'are'} dark: the server ` +
    `answering here is running older code and has no route for ${n === 1 ? 'it' : 'them'}. ` +
    `Restarting the dashboard from a terminal lights ${n === 1 ? 'it' : 'them'} up — not from a ` +
    `background daemon, which macOS can never grant camera access to.`
  )
}
