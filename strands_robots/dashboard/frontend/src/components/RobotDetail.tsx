import { useEffect, useMemo, useRef, useState } from 'react'
import { cameraEvidence, cameraPlaceholder } from '../lib/cameraEvidence'
import { useDialogFocus } from '../lib/useDialogFocus'
import type { Peer, StreamStep } from '../types'
import { useTask } from '../lib/useTask'
import { twinButtonCopy } from '../lib/twinButton'
import { statusSentence, peerStatusFields } from '../lib/statusSentence'
import { teleopView, stopVerdict, startVerdict, type TeleopView } from '../lib/teleopView'
import { leaderOptions, pairPlan, type PairInput } from '../lib/teleopPair'
import { api } from '../lib/endpoints'
import { useTelemetry } from '../lib/useTelemetry'
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
export default function RobotDetail({ peer, twinLive = false, hostsChildren, fleet, onClose }: {
  peer: Peer
  /** a '<id>-twin' peer is live in the fleet */
  twinLive?: boolean
  /** Q150: children this peer hosts, when it is a process rather than an arm. */
  hostsChildren?: string[] | null
  /** U22: the fleet's joint counts + measured roles, for "who could lead this arm". */
  fleet?: PairInput[] | null
  onClose: () => void
}) {
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin } = useTask(peer)
  const cams = Object.keys(peer.cameras ?? {})
  // R2: same words as the card, from the same pure module.
  const twin = twinButtonCopy({ peerId: peer.peer_id, twinLive, busy: twinBusy })
  const [cam, setCam] = useState<string | null>(null)
  /* U22 slice 1, READ-ONLY. Teleop has four server routes and no screen at all, so the dashboard has
     been telling operators to "collect teleop episodes" while giving them no way to see whether teleop
     is even working. The server's verdict already exists (teleop_health.py) and was written because of
     a measured disaster: 176 frames published, every one refused, while every surface the dashboard
     could see said success. Fetched ON DEMAND rather than polled — asking a peer costs a mesh
     round-trip on the same shared servo bus that starves the state reads. */
  const [teleop, setTeleop] = useState<TeleopView | null | 'asking' | 'unreachable'>(null)
  const askTeleop = async () => {
    setTeleop('asking')
    try { setTeleop(teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`))) }
    // A failed ASK is not an idle arm: say the ask failed. endpoints already explains a 404 from an
    // older server, and swallowing this into "no teleop" would be the same lie the counters told.
    catch { setTeleop('unreachable') }
  }
  /* U22 slice 2. STOPPING is the safe direction — it can only remove commands from an arm — so it
     needs no consent, but it does need (a) an armed two-step, because a mis-click during a good
     recording session costs the operator the take, and (b) a MEASURED result: the sentence afterwards
     comes from asking again, never from the POST returning 200. */
  const [stopArmed, setStopArmed] = useState(false)
  const [stopped, setStopped] = useState<{ ok: boolean; line: string } | null>(null)
  /* U22 slice 3b: STARTING. This is the only button on this screen whose effect is an arm in motion, so:
     armed-then-confirmed with the confirm sentence naming BOTH arms and which one moves; publish on the
     leader FIRST (read-only on that arm — it publishes what it measures) and only then point the follower
     at it, because a follower aimed at a stream nobody publishes waits out its subscribe budget and
     shrugs; and the result is MEASURED by asking again, where "started but every frame refused" is the
     outcome this fleet has actually produced. */
  const [startArmed, setStartArmed] = useState<string | null>(null)
  const [started, setStarted] = useState<{ ok: boolean; line: string } | null>(null)
  const startTeleop = async (leaderId: string) => {
    setStartArmed(null); setStarted(null); setStopped(null); setTeleop('asking')
    try {
      await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/publish`, { method: 'POST', body: JSON.stringify({}) })
    } catch (e) {
      setTeleop('unreachable'); setStarted({ ok: false, line: `${leaderId} could not start publishing, so nothing was pointed at it: ${(e as Error).message}` }); return
    }
    try {
      await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/receive`, { method: 'POST', body: JSON.stringify({ source_peer_id: leaderId }) })
    } catch (e) {
      /* HALF-BUILT CHAIN: the leader IS publishing now and the follower refused. Naming the remedy in
         prose was not enough — the arm to stop is a DIFFERENT peer than the one on this screen, so acting
         on the advice meant leaving the failure behind to go find it. The button comes with the sentence. */
      setTeleop('unreachable'); setStranded(leaderId)
      setStarted({ ok: false, line: `${leaderId} is publishing its joints, but ${peer.peer_id} would not follow it: ${(e as Error).message} — nothing is moving, and ${leaderId} is still on the wire until you stop it below` })
      return
    }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`)) } catch { after = null }
    setTeleop(after ?? 'unreachable')
    setStarted(startVerdict(after))
  }

  /* The leader left publishing by a half-built start, and its own stop — measured the same way as every
     other stop on this screen: ASK AGAIN, because "stop was sent" is not "it stopped". The result is kept
     even once it succeeds (a result that disappears when it works is the defect iter 485 fixed). */
  const [stranded, setStranded] = useState<string | null>(null)
  const [strandedResult, setStrandedResult] = useState<{ ok: boolean; line: string } | null>(null)
  const stopStranded = async (leaderId: string) => {
    setStrandedResult(null)
    try { await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop/stop`, { method: 'POST' }) }
    catch (e) {
      setStrandedResult({ ok: false, line: `${leaderId} would not stop publishing: ${(e as Error).message} — it is still on the wire` }); return
    }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(leaderId)}/teleop`)) } catch { after = null }
    const v = stopVerdict(after)
    setStrandedResult({ ok: v.ok, line: `${leaderId}: ${v.line}` })
  }

  const stopTeleop = async () => {
    setStopArmed(false); setStopped(null); setTeleop('asking')
    try { await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop/stop`, { method: 'POST' }) }
    catch (e) { setTeleop('unreachable'); setStopped({ ok: false, line: `stop was refused: ${(e as Error).message}` }); return }
    let after: TeleopView | null = null
    try { after = teleopView(await api(`/api/robots/${encodeURIComponent(peer.peer_id)}/teleop`)) } catch { after = null }
    setTeleop(after ?? 'unreachable')
    setStopped(stopVerdict(after))
  }
  /* Q58: focus must land inside an overlay and go back to whatever opened it. */
  const sheetRef = useRef<HTMLElement | null>(null)
  useDialogFocus(sheetRef)
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
  const telemetry = useTelemetry(peer)
  // Q151: THE SAFETY SENTENCE BELONGS HERE MOST. This is the surface an operator has open while
  // walking up to the arm, and it said nothing about whether the stillness on screen is measured —
  // while the card behind it said "safe to approach" or refused to. Same pure rule, same fields.
  const status = (p?.robot_type ?? '?') === 'robot'
    ? statusSentence(peerStatusFields(peer, telemetry, hostsChildren))
    : null

  // Steps per second, measured off the stream itself rather than trusted from
  // the policy's declared control frequency.
  const stepHz = useMemo(() => {
    if (steps.length < 2) return 0
    const span = steps[0].t - steps[steps.length - 1].t
    return span > 0 ? (steps.length - 1) / span : 0
  }, [steps])

  return (
    <div className="detail-backdrop" onClick={onClose}>
      <section ref={sheetRef} className={`detail${offline ? ' stale' : ''}`}
               role="dialog" aria-label={`Robot ${peer.peer_id}`} onClick={e => e.stopPropagation()}>
        <header className="detail-head">
          <span className={`typebadge ${p?.robot_type ?? '?'}`}>{p?.robot_type ?? '?'}</span>
          <h2>{peer.peer_id}</h2>
          {/* Q164: three states told apart by COLOUR ALONE said nothing to a screen reader and
              nothing under forced colours. The words are the same ones RobotCard uses, and role
              makes aria-label announceable on a span that has no text of its own. */}
          <span className={offline ? 'dot off' : running ? 'dot busy' : 'dot on'} role="img"
                aria-label={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'}
                title={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'} />
          {p?.hostname && <span className="host">{p.hostname}</span>}
          {p?.robot_type === 'robot' && !peer.peer_id.endsWith('-twin') && (
            <button className={`twinbtn${twin.cls ? ` ${twin.cls}` : ''}`} onClick={toggleTwin}
                    disabled={twinBusy} title={twin.title} aria-label={twin.aria}
                  aria-pressed={twin.pressed}>{twin.label}</button>
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
          <button className="btn ghost" onClick={askTeleop} disabled={teleop === 'asking'}
                  title="is this arm following another arm, or publishing its own joints? (reads only)">
            {teleop === 'asking' ? 'asking…' : 'teleop'}
          </button>
          <button className="btn ghost" onClick={onClose} aria-label="close this robot" title="Escape">✕</button>
        </header>
        {teleop === 'unreachable' && (
          <p className="hint warn" role="status">could not ask this arm about teleop — it may be busy or gone; its own log (devices › logs) is where a refusal appears</p>
        )}
        {teleop && typeof teleop === 'object' && (
          <div className={`hint ${teleop.tone === 'warn' ? 'warn' : ''}`} role="status">
            <b>teleop:</b> {teleop.headline}
            {teleop.streaming && <span className="muted small"> · frames are on the wire</span>}
            {teleop.detail && <div className="muted small">{teleop.detail}</div>}
            {/* The envelope is a SAFETY bound: the screen names the consent that widens it and never
                widens it here. ConsentSettings already renders this kind. */}
            {teleop.streaming && (
              stopArmed
                ? (
                  <div className="row small">
                    <button className="btn danger" onClick={stopTeleop}>confirm — stop teleop on {peer.peer_id}</button>
                    <button className="btn ghost" onClick={() => setStopArmed(false)}>keep it running</button>
                  </div>
                )
                : <button className="btn ghost small" onClick={() => setStopArmed(true)}
                          title="stop the teleop stream on this arm — it only removes commands, it cannot move anything">
                    stop teleop
                  </button>
            )}
            {/* U22 slice 3a: what would it TAKE to teleop this arm? Answered from evidence the peers
                carry, and only when the arm is not already streaming. The refusals matter more than the
                offers here: on this fleet both real arms report no joints, which is why teleop has never
                started — a fact that until now lived only in a child log. */}
            {!teleop.streaming && (fleet?.length ?? 0) > 0 && (() => {
              const opts = leaderOptions(peer.peer_id, fleet!)
              const usable = opts.filter(o => o.ok)
              if (usable.length) {
                const plan = pairPlan(peer.peer_id, usable[0].peer_id, fleet!)
                return (
                  <div className="small muted">
                    could follow: {usable.map(o => `${o.peer_id} (${o.why})`).join(', ')}
                    {plan && !plan.blockers.length && (
                      <> · starting asks for {plan.consents.join(' + ')} first, because frames move a real arm</>
                    )}
                    {plan?.notes.map((n, i) => <div key={i}>{n}</div>)}
                    {/* One armed step per candidate leader. The confirm sentence says which arm MOVES —
                        "start teleop" alone does not tell an operator standing next to two arms which of
                        them is about to travel. */}
                    <div className="row small">
                      {usable.map(o => (startArmed === o.peer_id ? (
                        <span key={o.peer_id} className="row small">
                          <button className="btn danger" onClick={() => startTeleop(o.peer_id)}>
                            confirm — hand-guide {o.peer_id}, and {peer.peer_id} MOVES with it
                          </button>
                          <button className="btn ghost" onClick={() => setStartArmed(null)}>cancel</button>
                        </span>
                      ) : (
                        <button key={o.peer_id} className="btn ghost small" onClick={() => setStartArmed(o.peer_id)}
                                title={`${peer.peer_id} will follow ${o.peer_id}'s joints and move`}>
                          follow {o.peer_id}
                        </button>
                      )))}
                    </div>
                  </div>
                )
              }
              return (
                <div className="small muted">
                  no arm on this fleet can lead it yet:
                  {opts.map(o => <div key={o.peer_id}>· {o.peer_id} — {o.why}</div>)}
                </div>
              )
            })()}
            {teleop.consentKind && (
              <div className="muted small">every frame is outside the safety envelope — settings › consent › {teleop.consentKind} is where that bound is widened, deliberately and by you</div>
            )}
          </div>
        )}
        {/* THE VERDICT ABOUT MY OWN ACTION MUST NOT LIVE INSIDE THE ARM'S STATUS BLOCK. It did, and that
            block only renders when the status could be RE-READ — so every failure path hid its own report:
            a stop that was refused, a start that was refused, and a half-built chain that left the leader
            publishing all end with the status 'unreachable', which is exactly when the operator most needs
            the sentence. Sibling of the status now, not a child of it. (Found by scenario 4 rendering an
            empty screen where a three-line explanation should have been.) */}
        {stopped && (
          <div className={`small ${stopped.ok ? 'muted' : 'warn'}`} role="status">{stopped.line}</div>
        )}
        {/* The result of a START must OUTLIVE the state it created. Rendered inside the "not streaming"
            offer, this line vanished at the exact moment it had something to say — a success the
            operator never saw, and a refusal (started, every frame rejected) hidden behind the very
            streaming flag that made it true. Found by the audit the moment it could render a fleet. */}
        {started && (
          <div className={`small ${started.ok ? 'muted' : 'warn'}`} role="status">{started.line}</div>
        )}
        {stranded && (
          <div className="row small">
            <button className="btn ghost small" onClick={() => stopStranded(stranded)}
                    disabled={strandedResult?.ok} title={`stop ${stranded} publishing its joints`}>
              stop {stranded} publishing
            </button>
            {strandedResult && (
              <span className={strandedResult.ok ? 'muted' : 'warn'} role="status">{strandedResult.line}</span>
            )}
          </div>
        )}
        {/* Q151: the safety sentence belongs on THIS surface most — it is what an operator reads while
            walking up to the arm. Same pure rule and same fields as the card, so the two cannot say
            different things about the same robot. */}
        {status && (
          <div className={`status-ribbon ${status.severity}`} role="status">
            <b>{status.word}</b> {status.text}
          </div>
        )}

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
              : (() => {
                  // "this peer publishes none" was a denial of the presence THIS SAME peer
                  // announces (lib/cameraEvidence): on a machine where macOS blocks capture,
                  // both arms announce top+wrist and deliver nothing, and the detail screen is
                  // where the operator comes to find out why. Say which of the two it is.
                  const ph = cameraPlaceholder(cameraEvidence(peer.peer_id, peer.presence?.cameras, cams, peer.cameras_requested))
                  return (
                    <div className="camtile big">
                      <div className="camstate" title={ph?.title}>
                        <b>{ph?.head ?? 'no camera'}</b><span>{ph?.sub ?? ''}</span>
                      </div>
                    </div>
                  )
                })()}
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
              // Q60: without this the detail screen's confirm sheet could not name the hardware it
              // was warning about, and showed a physical-motion warning for sim peers.
              presence={p}
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
            <JointStrip state={peer.state} presence={p} problem={peer.joint_problem} peerStale={peer.stale} />
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
