import { useState } from 'react'
import type { EstopResult } from '../types'
import { post, HttpError } from '../lib/endpoints'
import { estopFailureVerdict, resumeFailureVerdict, type FailureVerdict } from '../lib/estopOutcome'

/**
 * Fleet-wide stop, with a per-peer answer.
 *
 * The old flow fired the broadcast and showed nothing: a peer that never
 * answered, or that answered "cannot stop", looked exactly like a peer that
 * halted. For an e-stop that is the worst possible failure mode, so this stays
 * open until every live peer is accounted for, and stays *red* while any peer is
 * unconfirmed.
 */
export default function EstopSheet({
  open, onClose, linkWarning,
}: {
  open: boolean
  onClose: () => void
  /** Set when this page cannot currently deliver the stop (lib/linkHealth). The
   *  person who just pressed a dashed STOP ALL is owed the reason HERE, before
   *  the second click, not in a toast after it fails. */
  linkWarning?: string | null
}) {
  const [firing, setFiring] = useState(false)
  const [result, setResult] = useState<EstopResult | null>(null)
  // The VERDICT, not the message: whether the stop may have fired is the thing
  // the operator has to act on, and only the status can answer that.
  const [error, setError] = useState<FailureVerdict | null>(null)
  const [code, setCode] = useState('')
  const [resuming, setResuming] = useState(false)
  const [resumeMsg, setResumeMsg] = useState<string | null>(null)

  const resume = async () => {
    if (!code.trim()) return
    setResuming(true); setResumeMsg(null)
    try {
      const r = await post<{ status?: string; error?: string }>('/api/safety/resume', { override_code: code })
      if (r.status === 'ok') { setResumeMsg('✓ lockout cleared — fleet accepting commands again'); setCode('') }
      else setResumeMsg(`✗ ${r.error ?? 'resume rejected'} (wrong code? brute-force cooldown?)`)
    } catch (e: any) {
      // A resume whose answer never came back MAY have cleared the lockout;
      // reporting "still locked" would be a guess about the fleet's state.
      setResumeMsg(resumeFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: e?.message ?? String(e),
      }).text)
    } finally {
      setResuming(false)
    }
  }

  const fire = async () => {
    setFiring(true); setError(null)
    try {
      setResult(await post<EstopResult>('/api/safety/estop'))
    } catch (e: any) {
      setError(estopFailureVerdict({
        status: e instanceof HttpError ? e.status : 0,
        message: e?.message ?? String(e),
      }))
    } finally {
      setFiring(false)
    }
  }

  if (!open) return null

  const unconfirmed = result
    ? result.counts.not_stopped + result.counts.no_answer
    : 0

  return (
    <div className="sheet-backdrop" onClick={result && unconfirmed > 0 ? undefined : onClose}>
      <div className={`sheet estop-sheet${unconfirmed > 0 ? ' danger' : ''}`} onClick={e => e.stopPropagation()}>
        <h2>🛑 Stop everything</h2>

        {!result && !error && (
          <>
            <p>
              Sends <code>{'{action: "stop"}'}</code> to every peer with a live heartbeat and
              reports what each one answered.
            </p>
            <p className="hint">
              Fires BOTH rails: per-peer stop commands (answered individually below) and the
              signed <code>strands/safety/estop</code> envelope, which engages a fleet-wide
              LOCKOUT — every listening peer refuses further commands until a resume with the
              operator override code. A peer that is wedged or fully off the mesh still needs
              the hardware e-stop.
            </p>
            {linkWarning && (
              <p className="hint warn">
                ⚠ {linkWarning} Pressing STOP ALL is still worth it — it is sent the moment the
                link returns — but do not wait for it: the arms’ power switch is the only brake
                that does not go through this page.
              </p>
            )}
            <p className="hint">
              tip: <kbd>.</kbd> opens this sheet from anywhere — it works even when a drawer
              or dialog is covering the button.
            </p>
            <div className="sheet-actions">
              <button className="btn danger big" onClick={fire} disabled={firing}>
                {firing ? 'stopping…' : 'STOP ALL ROBOTS'}
              </button>
              <button className="btn ghost" onClick={onClose} disabled={firing}>cancel</button>
            </div>
          </>
        )}

        {error && (
          <>
            {/* "Nothing was sent" was the old line for EVERY failure — including a
                lost answer, where the stop may well have landed. The verdict
                distinguishes them, because the two demand different next moves. */}
            <div className="result bad">{error.headline}</div>
            <p className="hint warn">{error.advice}</p>
            <div className="sheet-actions">
              <button className="btn danger" onClick={fire}>
                {error.retryRepeats ? 'send the stop again' : 'retry'}
              </button>
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}

        {result && (
          <>
            <div className={result.all_stopped ? 'result ok' : 'result bad'}>
              {result.all_stopped
                ? `✓ all ${result.counts.stopped} live peer(s) confirmed stopped`
                : `⚠ ${unconfirmed} of ${result.targeted.length} peer(s) NOT confirmed stopped`}
            </div>
            <ul className="estop-list">
              {Object.entries(result.stopped).map(([peer, info]) => (
                <li key={peer} className={info.state}>
                  <b>{peer}</b>
                  <span>{info.state === 'stopped' ? 'stopped'
                    : info.state === 'no_answer' ? 'no answer — may still be moving'
                    : `refused: ${typeof info.detail === 'string' ? info.detail : JSON.stringify(info.detail)}`}</span>
                </li>
              ))}
            </ul>
            {result.stale_skipped.length > 0 && (
              <p className="hint">
                skipped (no heartbeat, cannot be reached): <code>{result.stale_skipped.join(', ')}</code>
              </p>
            )}
            {result.targeted.length === 0 && <p className="hint">No live peers were on the mesh.</p>}

            {result.lockout_engaged && (
              <div className="resume-box">
                <div className="result bad">
                  🔒 fleet LOCKOUT engaged{result.signed_rail?.issuer ? ` (signed by ${result.signed_rail.issuer})` : ''} —
                  peers refuse all commands until resumed
                </div>
                <div className="resume-row">
                  <input
                    type="password"
                    placeholder="operator override code"
                    value={code}
                    onChange={e => setCode(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && resume()}
                    disabled={resuming}
                  />
                  <button className="btn go" onClick={resume} disabled={resuming || !code.trim()}>
                    {resuming ? '…' : 'resume fleet'}
                  </button>
                </div>
                {resumeMsg && <div className="hint">{resumeMsg}</div>}
                <p className="hint">
                  The code is verified locally and an HMAC proof is broadcast — the code itself
                  never crosses the wire. Set <code>STRANDS_MESH_OVERRIDE_CODE</code> identically
                  on every peer.
                </p>
              </div>
            )}
            {result.signed_rail && !result.signed_rail.signed && (
              <p className="hint warn">
                ⚠ signed rail unavailable ({result.signed_rail.error}) — only per-peer stops were
                sent; no fleet lockout is in place.
              </p>
            )}

            <div className="sheet-actions">
              {!result.all_stopped && <button className="btn danger" onClick={fire}>send again</button>}
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
