import { useState } from 'react'
import type { Peer } from '../types'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'

const PROVIDERS = ['mock', 'lerobot_local', 'lerobot_async', 'groot', 'cosmos3']

export default function RobotCard({ peer }: { peer: Peer }) {
  const [provider, setProvider] = useState('mock')
  const [instruction, setInstruction] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  const p = peer.presence
  const type = p?.robot_type ?? '?'
  const cams = Object.keys(peer.cameras ?? {})
  const taskStatus = peer.state?.task?.status ?? p?.task_status
  const running = taskStatus === 'running' || taskStatus === 'executing'

  const start = async () => {
    if (!instruction.trim()) return
    setBusy(true); setResult(null)
    try {
      const res = await fetch(`/api/robots/${peer.peer_id}/task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction, policy_provider: provider }),
      })
      const j = await res.json()
      setResult(JSON.stringify(j.result).slice(0, 120))
    } catch (e) { setResult(String(e)) }
    setBusy(false)
  }

  const stop = async () => {
    setBusy(true)
    try { await fetch(`/api/robots/${peer.peer_id}/stop`, { method: 'POST' }) } catch {}
    setBusy(false)
  }

  return (
    <div className={peer.stale ? 'card stale' : 'card'}>
      <div className="card-head">
        <span className={`typebadge ${type}`}>{type}</span>
        <span className="peername" title={peer.peer_id}>{peer.peer_id}</span>
        {p?.hostname && <span className="host">{p.hostname}</span>}
        <span className={peer.stale ? 'dot off' : 'dot on'} />
      </div>

      {cams.length > 0 && (
        <div className={cams.length > 1 ? 'cams multi' : 'cams'}>
          {cams.slice(0, 4).map(c => <CameraTile key={c} peerId={peer.peer_id} cam={c} />)}
        </div>
      )}

      <JointStrip state={peer.state} />

      {peer.stream && (
        <div className="streamline">
          step {peer.stream.step} · {peer.stream.policy || 'policy'} · {peer.stream.instruction?.slice(0, 40)}
        </div>
      )}

      <div className="controls">
        <select value={provider} onChange={e => setProvider(e.target.value)} disabled={busy}>
          {PROVIDERS.map(pr => <option key={pr} value={pr}>{pr}</option>)}
        </select>
        <input
          placeholder="pick up the red cube"
          value={instruction}
          onChange={e => setInstruction(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && start()}
          disabled={busy}
        />
        {running
          ? <button className="btn stop" onClick={stop} disabled={busy}>■</button>
          : <button className="btn go" onClick={start} disabled={busy || !instruction.trim()}>▶</button>}
      </div>

      {result && <div className="result">{result}</div>}
    </div>
  )
}
