/**
 * Is a 404 the server being OLDER than this bundle? (Q79 — the general case of Q78)
 *
 * The dashboard process runs for days; the PWA bundle is rebuilt on every landed change. Measured on
 * this Mac 2026-08-20 by diffing the running server's own /openapi.json against the routes in the
 * current source: EIGHT routes the shipped bundle calls did not exist on the running server —
 * /api/devices/camera/{index}/modes, /api/devices/{peer_id}/cameras, /api/checkpoints/features,
 * /api/deploy/snippet, /api/robots/{peer_id}/policy-fit, /api/network/hint, /api/training/output-dir,
 * /api/devices/spawn-remembered. Each one failed with a bare 404 that reads like the THING is missing
 * (no such camera, no such peer) when what is missing is the server's age. Q78 fixed one field by hand;
 * this fixes the whole class in one place, because the next feature will land in exactly this gap.
 *
 * The judgement needs the server's OWN route list (FastAPI publishes /openapi.json), because a 404 from
 * a route that DOES exist is real news about the resource and must be left alone.
 */

/** Turn "/api/devices/camera/{index}/modes" into a matcher for "/api/devices/camera/2/modes". */
function templateMatches(template: string, path: string): boolean {
  if (!template.includes('{')) return template === path
  const parts = template.split(/\{[^}]*\}/)
  const rx = new RegExp('^' + parts.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('[^/]+') + '$')
  return rx.test(path)
}

/** The path a request actually asked for, without query or fragment or a trailing slash. */
export function normalisePath(path: string): string {
  const bare = path.split('#')[0].split('?')[0]
  return bare.length > 1 && bare.endsWith('/') ? bare.slice(0, -1) : bare
}

/**
 * Does the running server route this path at all?
 *
 * `null` when we do not know (openapi.json not fetched, unreachable, or a server that does not publish
 * one) — and "do not know" must never be rendered as "too old": that would blame the server for our own
 * missing fetch, and the resource's real 404 would disappear behind a wrong explanation.
 */
export function routeKnown(livePaths: string[] | null | undefined, path: string): boolean | null {
  if (!Array.isArray(livePaths) || livePaths.length === 0) return null
  const wanted = normalisePath(path)
  return livePaths.some(t => templateMatches(t, wanted))
}

/**
 * The sentence for a 404 whose route the server does not have. Names the path (so a bug report can be
 * acted on) and the remedy, which is the only thing that helps: this server is running older code.
 */
export function staleRouteMessage(path: string): string {
  return (
    `this dashboard's server does not have ${normalisePath(path)} — it is running code from before ` +
    `this feature existed. Restart the dashboard from a terminal to pick it up (the page itself is ` +
    `already new; a restart from an agent or daemon would come back with no camera access).`
  )
}

/**
 * The server ADMITTING, in its own 404 body, that it routes nothing at this path.
 *
 * routeKnown() is the authoritative test and is preferred, but it needs /openapi.json — when that
 * cannot be read it returns null and this module deliberately stays silent rather than blame the
 * server for our own missing fetch. This is the second-best evidence, and it comes from the server
 * too: an unrouted /api path falls through to the dashboard's SPA catch-all, which answers
 * {"error":"not found","detail":"no endpoint at /api/…"} (server.py ~2117) precisely so a missing
 * endpoint cannot be served index.html and look like a 200.
 *
 * ONLY THE DETAIL IS TRUSTED. `error: "not found"` looks like a discriminator and is not one: nothing
 * stops a route's OWN 404 from carrying that same word, and treating a genuine resource 404 ("no
 * dataset directory at /tmp/x") as staleness sends an operator to restart a server that is working
 * correctly. Measured against the live dashboard, because the stock framework wording ("Not Found")
 * is NOT what this app sends — a matcher written from the framework's docs was dead code here.
 */
export function unroutedByDetail(detail: unknown): boolean {
  if (typeof detail !== 'string') return false
  const d = detail.trim().toLowerCase()
  return d.startsWith('no endpoint at') || d === 'not found'
}
