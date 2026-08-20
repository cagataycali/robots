import { useEffect, useRef, useState } from 'react'
import type { Peer, StopResult } from '../types'
import { HttpError, post } from './endpoints'
import { findConsent, type ConsentNeed } from './consent'
import { runFailure, stopFailure } from './taskOutcome'
import type { RunBody } from '../components/RunForm'

/**
 * What we believe about this robot's task.
 *
 * `starting` and `stopping` exist because the mesh RPC is not instant: without
 * them the ▶ button re-enables while the command is still in flight, and a
 * double tap sends the task twice.
 */
export type TaskPhase = 'idle' | 'starting' | 'running' | 'stopping' | 'failed' | 'done'

export interface Outcome {
  ok: boolean
  text: string
  detail?: string
  /** The request failed WITHOUT telling us whether it landed: the arm may be
   *  moving (run) or may still be moving (stop). Rendered louder than a plain
   *  refusal, because the two demand different behaviour from a human standing
   *  next to the hardware. */
  ambiguous?: boolean
}

/**
 * Run/stop for one peer, shared by the card and the detail view so both report
 * the *same* verdict. A response can arrive and still be a refusal
 * (`{error: …}` or `ok: false`), and a stop that got no answer is not a stop -
 * collapsing either into "started"/"stopped" is the most dangerous thing this
 * UI can do.
 */
export function useTask(peer: Peer) {
  const [phase, setPhase] = useState<TaskPhase>('idle')
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [twinBusy, setTwinBusy] = useState(false)
  // A refusal the operator can answer (U18) plus the body that was refused, so
  // "approve" can re-send the SAME request instead of asking them to retype it.
  const [consent, setConsent] = useState<ConsentNeed | null>(null)
  const lastBody = useRef<RunBody | null>(null)
  const mounted = useRef(true)
  useEffect(() => () => { mounted.current = false }, [])

  const reported = peer.state?.task?.status ?? peer.presence?.task_status
  const reportedRunning = reported === 'running' || reported === 'executing'
  // The peer's own status wins over our optimistic phase - it is the robot
  // telling us what it is doing, we are only guessing.
  const running = reportedRunning || phase === 'starting' || phase === 'running'
  const busy = phase === 'starting' || phase === 'stopping' || twinBusy

  useEffect(() => {
    if (reportedRunning && phase !== 'running') setPhase('running')
    if (!reportedRunning && phase === 'running') setPhase('done')
  }, [reportedRunning])   // eslint-disable-line react-hooks/exhaustive-deps

  /** Bare message - only for requests with nothing physical behind them. */
  const fail = (e: unknown): Outcome => {
    if (e instanceof HttpError) {
      return { ok: false, text: e.message, detail: e.status ? `HTTP ${e.status}` : 'unreachable' }
    }
    return { ok: false, text: e instanceof Error ? e.message : String(e) }
  }

  /**
   * A thrown run/stop is NOT proof the robot was left alone: a rejected fetch
   * covers "never left this machine" and "dispatched, then lost the answer", and
   * a 5xx means the handler ran. lib/taskOutcome states which world it is and
   * what to do about it - see that file for why "failed" was the dangerous word.
   */
  const physicalFail = (e: unknown, kind: 'run' | 'stop'): Outcome => {
    const f = { status: e instanceof HttpError ? e.status : 0, message: e instanceof Error ? e.message : String(e) }
    const v = kind === 'run' ? runFailure(f) : stopFailure(f)
    return { ok: false, text: v.text, detail: v.detail, ambiguous: v.ambiguous }
  }

  const run = async (body: RunBody) => {
    setPhase('starting'); setOutcome(null); setConsent(null)
    lastBody.current = body
    try {
      const res = await post<{ ok: boolean; result: any; routed_to?: string; mirrored_to_twin?: boolean }>(
        `/api/robots/${encodeURIComponent(peer.peer_id)}/task`, body,
      )
      if (!mounted.current) return
      const err = res.result?.error ?? res.result?.result?.error
      setOutcome(res.ok
        ? { ok: true, text: `running${res.routed_to ? ` via ${res.routed_to}` : ''}${res.mirrored_to_twin ? ' + twin' : ''}` }
        : { ok: false, text: err ? String(err) : 'refused', detail: JSON.stringify(res.result).slice(0, 300) })
      setPhase(res.ok ? 'running' : 'failed')
      if (!res.ok) setConsent(findConsent(res))
    } catch (e) {
      if (!mounted.current) return
      setOutcome(physicalFail(e, 'run')); setPhase('failed')
      // A 4xx carries the same needs_consent in its body - a validation refusal
      // must be as answerable as a peer's refusal.
      if (e instanceof HttpError) setConsent(findConsent(e.body))
    }
  }

  /** Re-send the exact request that was refused (after a grant). */
  const retryLast = async () => {
    setConsent(null)
    if (lastBody.current) await run(lastBody.current)
  }

  const stop = async () => {
    setPhase('stopping'); setOutcome(null)
    try {
      const res = await post<StopResult>(`/api/robots/${encodeURIComponent(peer.peer_id)}/stop`)
      if (!mounted.current) return
      // stopped / not_stopped / no_answer - never a bare "stopped" on silence.
      const text = res.state === 'stopped' ? 'stopped'
        : res.state === 'no_answer' ? 'no answer - robot may still be moving'
        : `not stopped: ${typeof res.detail === 'string' ? res.detail : JSON.stringify(res.detail ?? {})}`
      setOutcome({ ok: res.state === 'stopped', text })
      setPhase(res.state === 'stopped' ? 'idle' : 'failed')
    } catch (e) {
      if (!mounted.current) return
      setOutcome(physicalFail(e, 'stop')); setPhase('failed')
    }
  }

  const toggleTwin = async () => {
    setTwinBusy(true)
    try {
      await post(`/api/robots/${encodeURIComponent(peer.peer_id)}/twin`, {})
    } catch (e) {
      setOutcome(fail(e))
    } finally {
      if (mounted.current) setTwinBusy(false)
    }
  }

  return {
    phase, outcome, running, busy, twinBusy, run, stop, toggleTwin, setOutcome,
    consent, clearConsent: () => setConsent(null), retryLast,
  }
}
