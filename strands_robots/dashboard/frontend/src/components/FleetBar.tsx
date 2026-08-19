import type { MeshInfo } from '../types'
import type { ConnState } from '../lib/useMesh'
import { backendLabel } from '../lib/endpoints'
import StrandsMark from './StrandsMark'

const CONN_LABEL: Record<ConnState, string> = {
  open: 'LIVE',
  connecting: 'CONNECTING',
  closed: 'OFFLINE',
  unauthorized: 'NO ACCESS',
}

interface Props {
  conn: ConnState
  peerCount: number
  dashboardId: string
  safetyFlash: string | null
  mesh: MeshInfo
  online: boolean
  installable: boolean
  activityCount: number
  onInstall: () => void
  onEstop: () => void
  onSettings: () => void
  onActivity: () => void
  onDevices: () => void
  onTraining: () => void
}

export default function FleetBar({
  conn, peerCount, dashboardId, safetyFlash, mesh, online, installable,
  activityCount, onInstall, onEstop, onSettings, onActivity, onDevices, onTraining,
}: Props) {
  // The mesh session and this browser's socket fail independently: the page can
  // be LIVE while the robot mesh is down, and vice versa. Showing only one of
  // them is how "why is the fleet empty" becomes unanswerable.
  const meshDown = mesh.online === false

  return (
    <header className="fleetbar">
      <div className="brand">
        <span className="logo"><StrandsMark size={26} title="Strands Agents" /></span>
        <div>
          <h1>strands robots</h1>
          <div className="sub" title={`API: ${backendLabel()}`}>
            {dashboardId || 'fleet cockpit'}
            <span className="backend"> · {backendLabel()}</span>
          </div>
        </div>
      </div>

      <div className="fleet-right">
        {safetyFlash && (
          <span className={`safety ${safetyFlash}`}>
            {safetyFlash === 'estop' ? '🛑 E-STOP' : '✅ RESUMED'}
          </span>
        )}
        {!online && <span className="badge warn" title="this device has no network">offline</span>}
        {mesh.local_dev && (
          <span className="badge danger" title="STRANDS_MESH_LOCAL_DEV=1 - mesh traffic is unauthenticated and unencrypted">
            wire security off
          </span>
        )}
        {meshDown && <span className="badge danger" title="the dashboard's own mesh session is closed">mesh down</span>}

        {installable && (
          <button className="chip" onClick={onInstall} title="Install as an app">⤓ install</button>
        )}
        <button className="chip" onClick={onDevices} title="Local hardware and managed robots">⚙ devices</button>
        <button className="chip" onClick={onTraining} title="Train policies on recorded datasets">🎓 train</button>
        <button className="chip" onClick={onActivity} title="Command history">
          ☰ activity{activityCount > 0 ? ` (${activityCount})` : ''}
        </button>
        <button className="chip" onClick={onSettings} title="Settings">⚒ settings</button>

        <span className="peers">{peerCount} peer{peerCount === 1 ? '' : 's'}</span>
        <span className={`conn ${conn}`} title={conn === 'unauthorized' ? 'the server rejected this token - set it in Settings' : ''}>
          {CONN_LABEL[conn]}
        </span>
        <button className="estop" onClick={onEstop} title="Stop every robot on the mesh">
          🛑 STOP ALL
        </button>
      </div>
    </header>
  )
}
