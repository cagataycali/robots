import { useEffect } from 'react'
import type { Peer } from '../types'
import { useTask } from '../lib/useTask'
import { useTelemetry } from '../lib/useTelemetry'
import { statusSentence } from '../lib/statusSentence'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'
import TelemetryStrip from './TelemetryStrip'
import RunForm from './RunForm'

export default function RobotCard({ peer, onOpen, onBusyChange }: {
  peer: Peer
  onOpen?: (peerId: string) => void
  onBusyChange?: (peerId: string, running: boolean) => void
}) {
  const { phase, outcome, running, busy, twinBusy, run, stop, toggleTwin } = useTask(peer)

  const p = peer.presence
  const type = p?.robot_type ?? '?'
  const cams = Object.keys(peer.cameras ?? {})
  const offline = !!peer.stale
  const telemetry = useTelemetry(peer)

  // The 5-second answer: one sentence joining heartbeat, hardware, task and
  // MEASURED motion - so "running but frozen" and "moving with no task"
  // (teleop/runaway: exactly when hands must stay clear) are said out loud
  // instead of being left for the operator to infer from four widgets.
  const status = type === 'robot' ? statusSentence({
    stale: offline,
    lastSeenAgoS: peer.last_seen ? Date.now() / 1000 - peer.last_seen : null,
    hwConnected: p?.connected ?? null,
    taskStatus: peer.state?.task?.status ?? p?.task_status ?? null,
    instruction: peer.state?.task?.instruction || p?.instruction || null,
    taskDurationS: peer.state?.task?.duration ?? null,
    moving: telemetry.moving,
    stateAgeS: telemetry.stateAgeS,
  }) : null

  // The app keeps a screen wake lock while anything is moving.
  useEffect(() => { onBusyChange?.(peer.peer_id, running) }, [running, peer.peer_id])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={`card${offline ? ' stale' : ''}${phase === 'failed' ? ' failed' : ''}${running ? ' running' : ''}`}>
      <div className="card-head">
        <span className={`typebadge ${type}`}>{type}</span>
        <button className="peername" title={`open ${peer.peer_id}`} onClick={() => onOpen?.(peer.peer_id)}>
          {peer.peer_id}
        </button>
        {p?.hostname && <span className="host">{p.hostname}</span>}
        {p?.connected === false && type === 'robot' && (
          <span className="badge warn" title="peer is online but its hardware is not connected">hw off</span>
        )}
        {type === 'robot' && !peer.peer_id.includes('__') && !peer.peer_id.endsWith('-twin') && (
          <button className="twinbtn" onClick={toggleTwin} disabled={twinBusy} title="Toggle sim twin">⿻</button>
        )}
        <span className={offline ? 'dot off' : running ? 'dot busy' : 'dot on'}
              title={offline ? 'no heartbeat for 15s' : running ? 'task running' : 'idle'} />
      </div>

      {status && (
        <div className={`status-ribbon ${status.severity}`} role="status">
          <b className="status-word">{status.word}</b>
          <span>{status.text}</span>
        </div>
      )}
      {offline && !status && (
        <div className="stale-note">
          no heartbeat — last seen{' '}
          {peer.last_seen ? `${Math.round(Date.now() / 1000 - peer.last_seen)}s ago` : 'unknown'}
        </div>
      )}

      {cams.length > 0 && (
        <div className={cams.length > 1 ? 'cams multi' : 'cams'}>
          {cams.slice(0, 4).map(c => (
            <CameraTile key={c} peerId={peer.peer_id} cam={c} meta={peer.cameras?.[c]} />
          ))}
        </div>
      )}

      <JointStrip state={peer.state} presence={p} />
      <TelemetryStrip peer={peer} />

      {peer.stream && (
        <div className="streamline">
          step {peer.stream.step} · {peer.stream.policy || 'policy'} · {peer.stream.instruction?.slice(0, 40)}
        </div>
      )}

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
          <span>{outcome.ok ? '✓' : '⚠'} {outcome.text}</span>
          {outcome.detail && <details><summary>details</summary><pre>{outcome.detail}</pre></details>}
        </div>
      )}
    </div>
  )
}
