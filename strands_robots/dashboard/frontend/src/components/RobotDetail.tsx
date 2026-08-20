import { useEffect, useMemo, useRef, useState } from 'react'
import type { Peer, StreamStep } from '../types'
import { useTask } from '../lib/useTask'
import { twinButtonCopy } from '../lib/twinButton'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'
import TelemetryStrip from './TelemetryStrip'
import RunForm from './RunForm'
import CameraConfigSheet from './CameraConfigSheet'

const HISTORY = 40

function fmt(v: unknown): string {
  if (typeof v === 'number') return v.toFixed(3)
  if (Array.isArray(v)) return `[${v.map(x => (typeof x === 'number' ? x.toFixed(2) : String(x))).join(', ')}]`
  if (v && typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/**
 * Single-robot stage: one big camera, the whole joint table, and the policy's
 * own step stream.
 *
 * The grid view is for watching a fleet; this is for watching *one* robot do
 * something, which is when you need the action vector the policy is actually
 * emitting rather than a 3-line summary. Steps are buffered client-side because
 * `strands/<peer>/stream` is fire-and-forget - nothing on the mesh replays it.
 */
export default function RobotDetail({ peer, twinLive = false, onClose }: {
  peer: Peer
  /** a '<id>-twin' peer is live in the fleet */
  twinLive?: boolean
  onClose: () => void
}) {
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin } = useTask(peer)
  const cams = Object.keys(peer.cameras ?? {})
  // R2: same words as the card, from the same pure module.
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy })
  const [cam, setCam] = useState<string | null>(null)
  const [camConfig, setCamConfig] = useState(false)
  const [steps, setSteps] = useState<StreamStep[]>([])
  const lastStep = useRef<number | null>(null)

  const active = cam && cams.includes(cam) ? cam : cams[0] ?? null

  useEffect(() => {
    const s = peer.stream
    if (!s || s.step === lastStep.current) return
    lastStep.current = s.step
    setSteps(prev => [s, ...prev].slice(0, HISTORY))
  }, [peer.stream])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const p = peer.presence
  const offline = !!peer.stale
  const joints = Object.entries(peer.state?.joints ?? {})

  // Steps per second, measured off the stream itself rather than trusted from
  // the policy's declared control frequency.
  const stepHz = useMemo(() => {
    if (steps.length < 2) return 0
    const span = steps[0].t - steps[steps.length - 1].t
    return span > 0 ? (steps.length - 1) / span : 0
  }, [steps])

  return (
    <div className="detail-backdrop" onClick={onClose}>
      <section className={`detail${offline ? ' stale' : ''}`} onClick={e => e.stopPropagation()}>
        <header className="detail-head">
          <span className={`typebadge ${p?.robot_type ?? '?'}`}>{p?.robot_type ?? '?'}</span>
          <h2>{peer.peer_id}</h2>
          <span className={offline ? 'dot off' : running ? 'dot busy' : 'dot on'} />
          {p?.hostname && <span className="host">{p.hostname}</span>}
          {p?.robot_type === 'robot' && !peer.peer_id.endsWith('-twin') && (
            <button className={`twinbtn${twin.cls ? ` ${twin.cls}` : ''}`} onClick={toggleTwin}
                    disabled={twinBusy} title={twin.title} aria-label={twin.aria}>{twin.label}</button>
          )}
          {p?.robot_type === 'robot' && peer.origin === 'external' && (
            <span className="originbadge"
                  title={'started outside this dashboard (your own script, or another machine).\n'
                    + 'Everything here works normally except the three things that need a local\n'
                    + 'child process: logs, camera reconfigure and despawn.'}>
              external
            </span>
          )}
          {p?.robot_type === 'robot' && (
            <button className="btn ghost" onClick={() => setCamConfig(true)}
                    /* U15: reconfiguring cameras IS a respawn, and we have no
                       process to respawn for a peer we did not start - the
                       request could only ever 404. Refuse it here, with the
                       reason, instead of opening a sheet that cannot submit.
                       Only when we KNOW: an absent origin (a server older than
                       the field) must keep the button working, and the sheet
                       already explains the 404 if it comes. */
                    disabled={peer.origin === 'external'}
                    title={peer.origin === 'external'
                      ? 'this robot was started outside the dashboard, so it has no local process to '
                        + 'restart — change its cameras where it was launched (its own script or machine)'
                      : 'attach / detach cameras, change fps and resolution (restarts the robot)'}>
              cameras
            </button>
          )}
          <button className="btn ghost" onClick={onClose} title="Escape">✕</button>
        </header>

        {offline && (
          <div className="stale-note">
            no heartbeat — last seen{' '}
            {peer.last_seen ? `${Math.round(Date.now() / 1000 - peer.last_seen)}s ago` : 'unknown'}.
            Commands will time out.
          </div>
        )}

        {camConfig && <CameraConfigSheet peerId={peer.peer_id} onClose={() => setCamConfig(false)} />}

        <div className="detail-body">
          <div className="detail-stage">
            {active
              ? <CameraTile peerId={peer.peer_id} cam={active} meta={peer.cameras?.[active]} big />
              : <div className="camtile big"><div className="camstate"><b>no camera</b><span>this peer publishes none</span></div></div>}
            {cams.length > 1 && (
              <div className="camswitch">
                {cams.map(c => (
                  /* aria-pressed, not colour alone: `.chip.on` is the ONLY thing that said which
                     camera is on screen, so a screen reader announced two identical "wrist, button"
                     controls and voice control could not confirm a switch. */
                  <button key={c} className={c === active ? 'chip on' : 'chip'}
                          aria-pressed={c === active} onClick={() => setCam(c)}>{c}</button>
                ))}
              </div>
            )}
            <TelemetryStrip peer={peer} />
            <RunForm
              peerId={peer.peer_id}
              running={running}
              busy={busy}
              disabled={offline}
              onRun={run}
              onStop={stop}
            />
            {outcome && (
              <div className={outcome.ok ? 'result ok' : 'result bad'}>
                <span>{outcome.ok ? '✓' : outcome.ambiguous ? '⚠ unknown —' : '✗'} {outcome.text}</span>
                {outcome.detail && <details><summary>details</summary><pre>{outcome.detail}</pre></details>}
              </div>
            )}
            {phase === 'stopping' && <div className="hint">stop sent, waiting for the peer to confirm…</div>}
          </div>

          <div className="side">
            <h3>Joints ({joints.length})</h3>
            <JointStrip state={peer.state} presence={p} />
            {joints.length > 0 && (
              <table className="jointtable">
                <thead><tr><th>joint</th><th>pos</th><th>vel</th></tr></thead>
                <tbody>
                  {joints.map(([name, v]) => {
                    const obj = v && typeof v === 'object' && !Array.isArray(v)
                      ? v as { position?: number; velocity?: number }
                      : null
                    const pos = obj?.position ?? (typeof v === 'number' ? v : Array.isArray(v) ? v[0] : undefined)
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="mono">{typeof pos === 'number' ? pos.toFixed(4) : '—'}</td>
                        <td className="mono">{obj?.velocity !== undefined ? obj.velocity.toFixed(3) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}

            <h3>Peer</h3>
            <dl className="kv">
              <dt>tool</dt><dd>{p?.tool_name ?? '—'}</dd>
              <dt>hardware</dt>
              <dd className={p?.connected === false ? 'bad' : ''}>
                {p?.connected === false ? 'not connected' : p?.hw ?? (p?.connected ? 'connected' : '—')}
              </dd>
              <dt>task</dt>
              <dd>{peer.state?.task?.status ?? p?.task_status ?? 'idle'}
                {peer.state?.task?.instruction ? ` — ${peer.state.task.instruction}` : ''}</dd>
              {p?.sim_robots?.length ? (
                <><dt>sim bodies</dt><dd className="mono">{p.sim_robots.join(', ')}</dd></>
              ) : null}
              {p?.action_keys?.length ? (
                <><dt>action keys</dt><dd className="mono">{p.action_keys.join(', ')}</dd></>
              ) : null}
              {p?.topics?.length ? (
                <><dt>topics</dt><dd className="mono small">{p.topics.join('\n')}</dd></>
              ) : null}
            </dl>

            <h3>
              Policy steps {steps.length > 0 && <em>{stepHz > 0 ? `${stepHz.toFixed(1)} Hz` : ''}</em>}
            </h3>
            {steps.length === 0 ? (
              <p className="hint">
                Nothing yet. Steps only arrive while a policy is running and only from the moment
                this view opened — the stream topic is not replayed.
              </p>
            ) : (
              <ul className="steps">
                {steps.map(s => (
                  <li key={`${s.step}-${s.t}`}>
                    <span className="stepno">#{s.step}</span>
                    <span className="mono">
                      {Object.entries(s.action ?? {}).slice(0, 8)
                        .map(([k, v]) => `${k}=${fmt(v)}`).join('  ') || '(no action)'}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
