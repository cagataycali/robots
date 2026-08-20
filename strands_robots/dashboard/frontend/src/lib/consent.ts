/**
 * The client half of U18: a refusal the operator can answer.
 *
 * The backend attaches `needs_consent` to a refused spawn or task (see
 * `strands_robots/dashboard/consent.py`). This module finds it wherever it
 * rode in — top level, nested under `result`, or inside an HttpError body from
 * a 422 — and decides what the dialog is allowed to offer.
 *
 * Pure except for `approveConsent`, which is one POST. Nothing here invents
 * copy: `title`, `risk` and `grants` come from the server, because the words
 * describing a security risk should be written once, next to the guard.
 */
import { post } from './endpoints'

export type ConsentKind = 'trust_remote_code' | 'hf_repo_allow' | string

export interface ConsentNeed {
  kind: ConsentKind
  scope: string
  title: string
  risk: string
  env_var?: string
  subject?: string | null
  grants?: string[]
  message?: string
}

export interface ConsentResult {
  granted: boolean
  already_granted?: boolean
  respawn_required?: boolean
  scope?: string
  grants?: string[]
  env_written?: string[]
  env_removed?: string[]
  note?: string
}

function looksLikeNeed(value: any): value is ConsentNeed {
  return (
    !!value &&
    typeof value === 'object' &&
    typeof value.kind === 'string' &&
    typeof value.scope === 'string' &&
    typeof value.title === 'string' &&
    typeof value.risk === 'string'
  )
}

/**
 * Dig `needs_consent` out of any shape the API answers with.
 *
 * Depth-limited on purpose: an unbounded walk over an arbitrary response is a
 * way to hang the UI on a cyclic object, and every real carrier is within two
 * levels (`body`, `body.result`, `body.detail`, `body.result.result`).
 */
export function findConsent(payload: any, depth = 3): ConsentNeed | null {
  if (!payload || typeof payload !== 'object' || depth <= 0) return null
  if (looksLikeNeed((payload as any).needs_consent)) return (payload as any).needs_consent
  for (const key of ['result', 'detail', 'body', 'error']) {
    const nested = (payload as any)[key]
    if (nested && typeof nested === 'object') {
      const found = findConsent(nested, depth - 1)
      if (found) return found
    }
  }
  return null
}

/**
 * Can this be approved at all?
 *
 * A refusal whose subject the server could not read safely (a hostile or
 * unparseable model name) is still worth showing — the operator should see what
 * was refused — but approving it would grant nothing, so the button must be
 * disabled rather than lying about having helped.
 */
export function canApprove(need: ConsentNeed): boolean {
  if (need.kind === 'hf_repo_allow') return !!need.subject
  return need.kind === 'trust_remote_code'
}

/** How dangerous is saying yes? Drives which button style the sheet uses. */
export function severity(need: ConsentNeed): 'danger' | 'warn' {
  // trust_remote_code grants arbitrary code execution, for every future load.
  return need.kind === 'trust_remote_code' ? 'danger' : 'warn'
}

/** The server owns the variable and the value; we send kind + subject only. */
export function approveConsent(need: ConsentNeed): Promise<ConsentResult> {
  return post<ConsentResult>('/api/consent', { kind: need.kind, subject: need.subject ?? null })
}

/**
 * What to tell the operator after a grant, and whether an immediate retry can
 * work. A peer that is already running kept the environment it was started
 * with, so for a peer-side refusal the honest next step is respawn-then-retry.
 */
export function afterApproval(
  result: ConsentResult,
  target: 'spawn' | 'peer',
): { retryNow: boolean; note: string } {
  const granted = result.granted || result.already_granted
  if (!granted) {
    return { retryNow: false, note: result.note || 'nothing was granted — retrying would fail the same way' }
  }
  if (target === 'spawn') {
    // The refused process never started, so the next spawn is the retry and it
    // inherits the grant from this dashboard's environment.
    return { retryNow: true, note: result.note || 'granted — starting again' }
  }
  return {
    retryNow: false,
    note:
      result.note ||
      'granted — but this robot is already running with the old permissions. Respawn it, then run again.',
  }
}
