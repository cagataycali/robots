/**
 * "That feature is broken" vs "the server process is older than this page".
 *
 * A loop that rebuilds the bundle without restarting the dashboard — which is the NORMAL state here,
 * because the camera-TCC law forbids an agent from bouncing that process (it can only be started
 * from a terminal-blessed shell, or it goes blind) — leaves the UI ahead of the server. Every
 * control added in that window calls a route the running process has never heard of, gets a 404, and
 * reports something that reads like a broken feature or a missing dataset. The operator's correct
 * action ("restart the dashboard") is nowhere in that sentence.
 *
 * THE DISCRIMINATOR IS THE DETAIL, NOT THE STATUS. Both of these are 404 on the same path:
 *   * an unrouted /api path — this dashboard's SPA catch-all answers
 *     {"error": "not found", "detail": "no endpoint at /api/…"} (server.py ~2117, deliberately, so a
 *     missing endpoint cannot be served index.html and look like a 200) — the route does not exist,
 *     so the server predates the page;
 *   * a route's OWN 404 ("no dataset directory at /x") — the route exists and answered about a
 *     resource.
 * Reading the second as skew would send someone to restart a server that is working, and reading the
 * first as a resource problem sends them hunting for a dataset that was never the issue.
 *
 * The shape above was MEASURED against the live dashboard, not assumed: the first version of this
 * file tested for FastAPI's stock detail "Not Found", which this app never sends — so the hint would
 * have been dead code in exactly the situation it exists for. Both shapes are accepted now, because
 * the stock one is what a bare TestClient app or a future refactor would produce.
 */

export type SkewProbe = { status?: number | null; detail?: unknown; error?: unknown; path?: string }

/** A sentence naming the real cause, or null when this failure is not version skew. */
export function skewHint(probe: SkewProbe | null | undefined): string | null {
  if (!probe || probe.status !== 404) return null
  const detail = typeof probe.detail === 'string' ? probe.detail.trim().toLowerCase() : ''
  // ONLY the detail is trusted. `error: "not found"` looks like a discriminator and is not one: this
  // app sends that envelope for its catch-all, and nothing stops a route's own 404 from carrying the
  // same word — matching on it flagged "no dataset directory at /tmp/x" as skew in the test below,
  // which would send an operator to restart a server that is working correctly.
  const unrouted = detail.startsWith('no endpoint at') || detail === 'not found'
  if (!unrouted) return null
  const what = probe.path ? ` (${probe.path})` : ''
  return (
    `this page is newer than the dashboard process it is talking to: the server has no such route${what}. ` +
    'Restart the dashboard from a terminal (not from an agent — it needs the terminal\'s camera grant) ' +
    'and this will work.'
  )
}

/** The message to show for a failed call: the skew explanation when that is what happened. */
export function failureText(probe: SkewProbe | null | undefined, fallback: string): string {
  return skewHint(probe) ?? fallback
}

/** Pull a probe out of whatever the fetch layer threw, without assuming its shape. */
export function probeFromError(e: unknown, path?: string): SkewProbe {
  const any = e as { status?: number; body?: { detail?: unknown; error?: unknown }; message?: string }
  return { status: any?.status ?? null, detail: any?.body?.detail ?? null, error: any?.body?.error ?? null, path }
}
