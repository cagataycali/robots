import { useState } from 'react'
import type { ConnState } from '../lib/useMesh'

export default function FleetBar({ conn, peerCount, dashboardId, safetyFlash }:
  { conn: ConnState; peerCount: number; dashboardId: string; safetyFlash: string | null }) {
  const [confirming, setConfirming] = useState(false)

  const estop = async () => {
    if (!confirming) { setConfirming(true); setTimeout(() => setConfirming(false), 3000); return }
    setConfirming(false)
    try { await fetch('/api/safety/estop', { method: 'POST' }) } catch {}
  }

  return (
    <header className="fleetbar">
      <div className="brand">
        <span className="logo">🤖</span>
        <div>
          <h1>strands robots</h1>
          <div className="sub">{dashboardId || 'fleet cockpit'}</div>
        </div>
      </div>
      <div className="fleet-right">
        {safetyFlash && <span className={`safety ${safetyFlash}`}>{safetyFlash === 'estop' ? '🛑 E-STOP' : '✅ RESUMED'}</span>}
        <span className="peers">{peerCount} peer{peerCount === 1 ? '' : 's'}</span>
        <span className={`conn ${conn}`}>{conn === 'open' ? 'LIVE' : conn.toUpperCase()}</span>
        <button className={confirming ? 'estop confirm' : 'estop'} onClick={estop}>
          {confirming ? 'CONFIRM STOP?' : '🛑 STOP ALL'}
        </button>
      </div>
    </header>
  )
}
