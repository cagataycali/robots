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
  /** Q120: the server's own answer to "would approving change anything". Absent on older servers. */
  grantable?: boolean
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
  // Q120: ASK THE SERVER. It owns env_patch, so it knows whether a grant exists here; the rule
  // below was right for the kinds that existed when it was written and wrong the moment Q119 added
  // two more allowlist kinds, offering an enabled button for a host the server could not read. A
  // boolean the guard computes cannot drift from the guard.
  if (typeof need.grantable === 'boolean') return need.grantable
  // Fallback for a server older than that field: a subject is required wherever the grant IS the
  // subject — every allowlist kind, not just the model repository.
  if (need.kind === 'hf_repo_allow' || need.kind === 'policy_type_allow'
      || need.kind === 'policy_host_allow') return !!need.subject
  // Otherwise: approvable when the server named something it would actually set. This used to be
  // `kind === 'trust_remote_code'` — a closed list of ONE — so the two guards added since
  // (teleop_degree_units, agent_physical_motion; consent.py KINDS has four) arrived with a complete,
  // grantable payload and got a disabled button plus "the name in this refusal could not be read
  // safely, check the model path". On this fleet that was the Q27 degree envelope: teleop refused
  // every frame, the dashboard offered the consent dialog, and the dialog could not grant it.
  // Asking the payload instead of a hardcoded list means the next guard works on arrival.
  return !!need.env_var || !!need.grants?.length
}

/**
 * Why approving is impossible, in the words of the actual reason.
 *
 * The sheet used to print one sentence for every blocked case ("the name could not be read safely …
 * check the model path"), which was false for any kind that has no model path — and false in a
 * SECURITY dialog, where a wrong explanation is what the operator carries away.
 */
export function blockedReason(need: ConsentNeed): string {
  if (need.kind === 'hf_repo_allow') {
    return 'The repository name in this refusal could not be read safely, so there is nothing to '
      + 'allow. Check the model path and try again.'
  }
  // Named separately because "check the model path" is false advice for these two, and a security
  // dialog's explanation is what the operator carries away (this function's own reason for existing).
  if (need.kind === 'policy_type_allow') {
    return 'The policy name in this refusal could not be read safely, so there is nothing to allow. '
      + 'Check the policy type or provider you asked for and try again.'
  }
  if (need.kind === 'policy_host_allow') {
    return 'The address in this refusal could not be read as a host, so there is nothing to allow. '
      + 'Use a plain hostname or IP (optionally with a port) and try again.'
  }
  return 'This refusal did not say what approving would change, so there is nothing to grant from '
    + 'here. It may come from a newer guard than this page — reload, and if it persists, grant it '
    + 'in the environment instead.'
}

/**
 * How dangerous is saying yes? Drives which button style the sheet uses.
 *
 * 'danger' is for a grant that is OPEN-ENDED — it hands over a capability, not a value:
 *   - trust_remote_code: arbitrary code execution, for every future policy load;
 *   - agent_physical_motion: the agent may start motion on any real robot on this mesh, unattended,
 *     from a chat sentence, with no confirmation step. It was styled as a mild warning, level with
 *     adding one repository to an allowlist.
 * 'warn' is for a bounded change: one allowlist entry, or an envelope that stays an envelope
 * (teleop_degree_units widens the bound and still refuses a runaway).
 * An UNKNOWN kind is treated as danger: a permission this page cannot recognise is the last thing
 * that should be presented as routine.
 */
const OPEN_ENDED: ReadonlySet<string> = new Set(['trust_remote_code', 'agent_physical_motion'])
// policy_type_allow is bounded in the same way hf_repo_allow is: one name added to one list.
// policy_host_allow is deliberately NOT here — it stays 'danger' by the unknown-kind rule, because
// approving it sends camera frames and joint states to another machine and lets what that machine
// returns drive the arms. That is a capability handed over, not a value widened.
const BOUNDED: ReadonlySet<string> = new Set(['hf_repo_allow', 'teleop_degree_units', 'policy_type_allow'])

export function severity(need: ConsentNeed): 'danger' | 'warn' {
  if (OPEN_ENDED.has(need.kind)) return 'danger'
  return BOUNDED.has(need.kind) ? 'warn' : 'danger'
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


/**
 * Does this machine grant anything beyond the SDK's own defaults?
 *
 * Q121. The permissions screen computed this inline as a conjunction naming each kind
 * (`!trust && hf.length === 0 && !envelope?.granted && !agent`), and that shape has now been wrong
 * three times: once when the teleop envelope arrived, once when agent motion did, and again when
 * Q119 added the two policy allowlists. Each time the screen said "Nothing extra is allowed here"
 * while a real grant was in force — the most damaging sentence a permissions page can get wrong,
 * because the operator reads it as an assurance.
 *
 * So this asks the PAYLOAD's own shape instead of a list of names: any truthy boolean, any non-empty
 * array, or a granted envelope means something is granted. A kind added by a newer server than this
 * bundle therefore counts on arrival, even though no row exists to display it yet — which is the
 * safe direction to be wrong in: the screen may under-explain, never falsely reassure.
 *
 * `locks` is skipped by name and it is the one exception worth hardcoding: it TIGHTENS the machine
 * (task_requires_confirm), so counting it as a grant would report a restriction as a permission.
 * `kinds` and `env_file` are metadata, not state.
 */
const NOT_A_GRANT: ReadonlySet<string> = new Set(['locks', 'kinds', 'env_file'])

export function nothingGranted(state: Record<string, unknown> | null | undefined): boolean {
  if (!state) return false  // unknown is not "nothing": say nothing rather than reassure
  for (const [key, value] of Object.entries(state)) {
    if (NOT_A_GRANT.has(key)) continue
    if (Array.isArray(value)) {
      if (value.length > 0) return false
    } else if (value && typeof value === 'object') {
      if ((value as { granted?: unknown }).granted) return false
    } else if (value === true) {
      return false
    }
  }
  return true
}
