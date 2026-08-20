/**
 * Does the SERVER this bundle is talking to accept a given form field? (Q78)
 *
 * The two halves of this app are routinely different ages: the dashboard process runs for days, the
 * PWA bundle is rebuilt whenever someone lands a change. Measured against the live dashboard on
 * 2026-08-20: a fresh form offered the new `val_episodes` holdout, the running server refused it with
 * "unknown field(s): val_episodes. Valid fields: provider, dataset_root, …" — true, useless, and the
 * real remedy (restart the dashboard) appears nowhere in it.
 *
 * /api/training/trainers now publishes `fields`. The rule here is deliberately asymmetric:
 *
 *   - list present and contains the key  -> accepted
 *   - list present and lacks the key     -> REFUSED here, with the remedy, before the click
 *   - list ABSENT (server predates the `fields` key itself) -> also refused for any field that
 *     shipped WITH or AFTER that key, which is sound rather than pessimistic: they are in the same
 *     release, so a server too old to publish the list is too old to accept the field.
 *   - the fetch has not happened or failed -> silent: never disable a field because of OUR network,
 *     that turns a hiccup into a missing feature. The server's own refusal is the fallback.
 */

export interface FieldSupport {
  ok: boolean
  /** why it cannot be used, as something the operator can DO — or '' when it can */
  why: string
}

export function fieldSupport(
  fields: string[] | null | undefined,
  key: string,
  loaded: boolean,
): FieldSupport {
  if (!loaded) return { ok: true, why: '' }
  if (Array.isArray(fields)) {
    if (fields.includes(key)) return { ok: true, why: '' }
    return {
      ok: false,
      why: `this dashboard's server does not accept ${key} — it is running code from before the field existed. Restart the dashboard to pick it up.`,
    }
  }
  return {
    ok: false,
    why: `this dashboard's server is older than ${key} (it does not publish the field list yet). Restart the dashboard to pick it up.`,
  }
}
