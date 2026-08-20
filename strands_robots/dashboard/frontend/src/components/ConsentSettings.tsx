/**
 * What this machine has been talked into allowing — and the way back (U18).
 *
 * The consent sheet tells the operator a grant can be revoked. That sentence is
 * only true if there is a place that lists the grants, so this is that place:
 * one row per permission, with the blast radius stated (an org-wide entry allows
 * every repo under it) and a revoke that is as narrow as the grant was.
 */
import { useCallback, useEffect, useState } from 'react'
import { api, post } from '../lib/endpoints'

type TeleopEnvelope = {
  granted: boolean
  value_abs: string | null
  slew_abs: string | null
  is_degree_preset: boolean
}

type ConsentState = {
  trust_remote_code: boolean
  hf_repo_allow: string[]
  /* Absent from an older server: this screen listed two of the three kinds and the teleop envelope
     widening — the grant with physical reach — could not be seen or revoked here. */
  teleop_degree_units?: TeleopEnvelope
  /* Q80: the agent's permission to START physical motion by itself. Absent from an older server. */
  agent_physical_motion?: boolean
  env_file?: string
}

export default function ConsentSettings() {
  const [state, setState] = useState<ConsentState | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setState(await api<ConsentState>('/api/consent'))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const revoke = async (kind: string, subject: string | null, label: string) => {
    setBusy(label); setNote(null); setError(null)
    try {
      const r = await post<{ revoked: boolean; note?: string }>('/api/consent/revoke', { kind, subject })
      setNote(r.note ?? (r.revoked ? `revoked ${label}` : `nothing to revoke for ${label}`))
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (error && !state) return <p className="hint">could not read permissions: {error}</p>

  const envelope = state?.teleop_degree_units
  const nothing = state && !state.trust_remote_code && state.hf_repo_allow.length === 0
    && !envelope?.granted && !state.agent_physical_motion

  return (
    <div className="consent-settings">
      <h3>Permissions you granted</h3>
      {nothing ? (
        <p className="hint">
          Nothing extra is allowed here. When a safety guard refuses something, the dashboard asks
          you first and what you approve shows up in this list.
        </p>
      ) : null}

      {state?.trust_remote_code ? (
        <div className="cg-row">
          <div>
            <b>Run model code from HuggingFace</b>
            <div className="hint">
              Any policy load may execute code from the model repository — this is the widest
              permission on the list.
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'trust'}
                  onClick={() => revoke('trust_remote_code', null, 'trust')}>
            {busy === 'trust' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {state?.agent_physical_motion ? (
        <div className="cg-row">
          <div>
            <b>The agent may start motion on real robots</b>
            <div className="hint">
              A chat sentence or a voice command can put any real robot on this mesh in motion, with no
              confirmation and without the check that the policy fits that robot — the one the ▶ button
              does. Revoking leaves the agent able to stop robots, answer questions and run tasks in
              simulation; starting a real arm comes back to you.
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'agent motion'}
                  onClick={() => revoke('agent_physical_motion', null, 'agent motion')}>
            {busy === 'agent motion' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {envelope?.granted ? (
        <div className="cg-row">
          <div>
            <b>Teleop envelope widened{envelope.is_degree_preset ? ' to degrees' : ''}</b>
            <div className="hint">
              Every teleop stream on this machine, not one arm: a single frame may command a reach
              of {envelope.value_abs ?? 'the default'} units
              {envelope.slew_abs ? ` at up to ${envelope.slew_abs} units/s` : ' (speed bound left at the default)'}.
              {envelope.is_degree_preset
                ? ' This is the degrees preset, which an SO-101 needs — a runaway far outside it is still refused.'
                : ' This is a hand-set bound, not the degrees preset — check it is the one you meant.'}
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === 'teleop'}
                  onClick={() => revoke('teleop_degree_units', null, 'the teleop envelope')}>
            {busy === 'teleop' ? '…' : 'revoke'}
          </button>
        </div>
      ) : null}

      {state?.hf_repo_allow.map(entry => (
        <div className="cg-row" key={entry}>
          <div>
            <b className="rc-mono">{entry}</b>
            <div className="hint">
              {entry.includes('/')
                ? 'this model only'
                : 'every model under this organisation — wider than a single approval'}
            </div>
          </div>
          <button className="btn ghost danger" disabled={busy === entry}
                  onClick={() => revoke('hf_repo_allow', entry, entry)}>
            {busy === entry ? '…' : 'revoke'}
          </button>
        </div>
      ))}

      {note ? <p className="cs-note">{note}</p> : null}
      {error ? <p className="cs-error">{error}</p> : null}
      <p className="hint">
        Revoking applies to robots started from now on: a peer that is already running keeps the
        permission it was started with until you respawn it.
      </p>
    </div>
  )
}
