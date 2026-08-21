import type { StopResult } from '../types'
import type { TaskPhase } from './taskPhase'

/**
 * Reading a run/stop RESPONSE — as pure functions.
 *
 * lib/taskOutcome already covers the THROWN cases (a rejected fetch, a 5xx: whether the arm may be
 * moving). This is the other half, which lived inside useTask's socket-less request handler and had
 * no test: a response that ARRIVED, and what it actually says.
 */

export interface Outcome {
  ok: boolean
  text: string
  detail?: string
  ambiguous?: boolean
}

/** Never throws and never returns undefined: this string is only ever shown to a human. */
function describe(value: unknown, cap = 300): string {
  if (value === undefined) return 'no result payload'
  try {
    return String(JSON.stringify(value) ?? String(value)).slice(0, cap)
  } catch {
    return String(value).slice(0, cap)
  }
}

/**
 * The error message a peer's response carries, at either nesting depth, or undefined.
 *
 * The wire is layered — the bridge's `result` is the peer's reply, and a peer that itself wraps a
 * tool result gives `result.result`. Both shapes are real, which is why this looks at both.
 */
export function errorInResult(result: any): string | undefined {
  const err = result?.error ?? result?.result?.error
  if (err === undefined || err === null || err === '') return undefined
  return String(err)
}

export interface RunResponse {
  ok?: boolean
  result?: any
  routed_to?: string
  mirrored_to_twin?: boolean
}

/**
 * Did the run actually start?
 *
 * A RESPONSE IS NOT A CONFIRMATION. Two independent things can say no, and the UI must honour
 * either one:
 *
 *  - `ok: false` — the bridge's verdict (mesh_bridge.command_succeeded).
 *  - an `error` inside the payload — the PEER's own words.
 *
 * They can disagree, and today they do (Q88): command_succeeded rejects a response for
 * `type == "error"`, a top-level `error`, `ok is False`, `result.ok is False` or a `result.status` of
 * error/failed — but NOT for `result.error`. So a peer reply of `{"result": {"error": "gripper
 * jammed"}}` — no ok key, no status — comes back as `ok: true` and used to be rendered "running",
 * because useTask extracted that nested error and then only consulted it when the envelope had
 * already said no. An error the operator can read, above a card claiming the task is running.
 *
 * Trusting the stricter of the two is the only safe direction here: claiming a task started when it
 * did not is what leaves someone waiting for an arm that will never move.
 */
export function interpretRun(res: RunResponse | undefined): { outcome: Outcome; phase: TaskPhase } {
  const err = errorInResult(res?.result)
  if (res?.ok && !err) {
    const via = res.routed_to ? ` via ${res.routed_to}` : ''
    const twin = res.mirrored_to_twin ? ' + twin' : ''
    return { outcome: { ok: true, text: `running${via}${twin}` }, phase: 'running' }
  }
  return {
    outcome: {
      ok: false,
      // The peer's own words beat "refused" — and when the envelope said ok while the payload
      // carried an error, that error IS the news.
      text: err ?? 'refused',
      detail: describe(res?.result),
    },
    phase: 'failed',
  }
}

/**
 * Did the stop actually stop it?
 *
 * stopped / not_stopped / no_answer — never a bare "stopped" on silence. A stop that got no answer
 * is not a stop, and an unrecognised state is not one either: this is the control a human reaches
 * for when an arm is doing something they do not like.
 */
export function interpretStop(res: StopResult | undefined): { outcome: Outcome; phase: TaskPhase } {
  const state = res?.state
  if (state === 'stopped') return { outcome: { ok: true, text: 'stopped' }, phase: 'idle' }
  if (state === 'no_answer') {
    return {
      outcome: { ok: false, text: 'no answer — robot may still be moving', ambiguous: true },
      phase: 'failed',
    }
  }
  // not_stopped, or a state this bundle does not recognise (an older/newer server). Both mean the
  // same thing to the person standing next to the arm: it was not stopped.
  const detail = typeof res?.detail === 'string' ? res.detail : describe(res?.detail ?? {})
  return { outcome: { ok: false, text: `not stopped: ${detail}` }, phase: 'failed' }
}
