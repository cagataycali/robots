import { useEffect, useMemo, useState } from 'react'
import { useMesh } from './lib/useMesh'
import { usePwa } from './lib/usePwa'
import { ConfigProvider } from './lib/useConfig'
import { backendKey, backendLabel, setAuthToken } from './lib/endpoints'
import FleetBar from './components/FleetBar'
import RobotCard from './components/RobotCard'
import RobotDetail from './components/RobotDetail'
import AgentDock from './components/AgentDock'
import SettingsDrawer from './components/SettingsDrawer'
import ActivityLog from './components/ActivityLog'
import DevicePanel from './components/DevicePanel'
import EstopSheet from './components/EstopSheet'
import TrainingTab from './components/TrainingTab'
import RecordPanel from './components/RecordPanel'
import AuthGate from './components/AuthGate'

type Panel = 'settings' | 'activity' | 'devices' | 'estop' | 'training' | 'record' | null

/**
 * `?panel=…` is what the manifest shortcuts deep-link to. Note there is
 * deliberately no `estop` shortcut in the manifest - a fleet stop needs the
 * confirm step and the per-peer results, which a launcher shortcut cannot show.
 */
function initialPanel(): Panel {
  const want = new URLSearchParams(location.search).get('panel')
  return want === 'settings' || want === 'activity' || want === 'devices' || want === 'training' || want === 'record' ? want : null
}

function Dashboard() {
  const { conn, dashboardId, peers, safetyFlash, mesh, activity, loaded } = useMesh()
  const pwa = usePwa()
  const [panel, setPanel] = useState<Panel>(initialPanel)
  const [detail, setDetail] = useState<string | null>(null)
  const [busyPeers, setBusyPeers] = useState<Record<string, boolean>>({})

  const list = useMemo(() => Object.values(peers)
    // Infrastructure peers are not robots: 'gateway' meshes (robot_mesh's
    // robot-less fallback, one per coordinating agent process) and the
    // dashboard's own '-safety' signer have no cameras, no joints, and no
    // dispatch - a run form on them is a card that can only fail. They stay
    // visible in the settings/mesh panel, just not in the fleet grid.
    .filter(p => {
      const t = p.presence?.robot_type
      if (t === 'gateway') return false
      if (p.peer_id.endsWith('-safety')) return false
      return true
    })
    .sort((a, b) => {
      // fresh first, then robots before sims, then name
      if (!!a.stale !== !!b.stale) return a.stale ? 1 : -1
      const at = a.presence?.robot_type ?? 'z', bt = b.presence?.robot_type ?? 'z'
      if (at !== bt) return at === 'robot' ? -1 : 1
      return a.peer_id.localeCompare(b.peer_id)
    }), [peers])

  const anyRunning = Object.values(busyPeers).some(Boolean)

  // A phone that sleeps mid-task drops the camera sockets, exactly when the
  // operator most needs to see a moving arm.
  useEffect(() => { void pwa.keepAwake(anyRunning) }, [anyRunning])  // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard: Escape closes whatever is on top, "." opens the stop sheet.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.tagName?.match(/INPUT|TEXTAREA|SELECT/)
      if (e.key === 'Escape') { setPanel(null); return }
      if (!typing && e.key === '.') setPanel('estop')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const detailPeer = detail ? peers[detail] : undefined
  useEffect(() => {
    // The peer vanished from the mesh entirely (never came back after a
    // re-point); a stage showing a peer that no longer exists is a lie.
    if (detail && !peers[detail]) setDetail(null)
  }, [detail, peers])

  return (
    <div className="stage">
      <FleetBar
        conn={conn}
        peerCount={list.filter(p => !p.stale).length}
        dashboardId={dashboardId}
        safetyFlash={safetyFlash}
        mesh={mesh}
        online={pwa.online}
        installable={pwa.installable}
        activityCount={activity.length}
        onInstall={() => void pwa.install()}
        onEstop={() => setPanel('estop')}
        onSettings={() => setPanel('settings')}
        onActivity={() => setPanel('activity')}
        onDevices={() => setPanel('devices')}
        onTraining={() => setPanel('training')}
        onRecord={() => setPanel('record')}
      />

      {pwa.needRefresh && (
        <div className="toast">
          A new dashboard version is ready.
          <button className="btn go" onClick={pwa.update}>reload</button>
          <span className="hint">Reloading drops camera streams; a running task keeps running.</span>
        </div>
      )}

      {!pwa.online && (
        <div className="toast warn">
          This device is offline — the fleet view is a cached snapshot and commands will fail.
        </div>
      )}

      {conn === 'unauthorized' ? (
        <div className="empty-fleet">
          <div className="empty-icon">🔒</div>
          <h2>This dashboard requires a token</h2>
          <p>
            <code>{backendLabel()}</code> refused the connection. Paste the token the server was
            started with:
          </p>
          <TokenPrompt />
          <p className="hint">
            It is the value of <code>--auth-token</code> / <code>DASHBOARD_AUTH_TOKEN</code>.
          </p>
        </div>
      ) : list.length === 0 ? (
        <div className="empty-fleet">
          <div className="empty-icon">{conn === 'open' ? '📡' : '🔌'}</div>
          {conn !== 'open' ? (
            <>
              <h2>Not connected to {backendLabel()}</h2>
              <p>
                {conn === 'connecting' ? 'Opening the mesh socket…' : 'The dashboard API is unreachable.'}
              </p>
              <p className="hint">
                If the API runs elsewhere, point this browser at it in Settings → Connection.
              </p>
              <button className="btn ghost" onClick={() => setPanel('settings')}>open settings</button>
            </>
          ) : mesh.online === false ? (
            <>
              <h2>The dashboard's mesh session is down</h2>
              <p>The API is up, but it is not on the robot mesh — so no peer can be seen or commanded.</p>
              <p className="hint">Check the mesh endpoints, then restart the session.</p>
              <button className="btn ghost" onClick={() => setPanel('settings')}>mesh settings</button>
            </>
          ) : (
            <>
              <h2>{loaded ? 'No robots on the mesh yet' : 'Loading the fleet…'}</h2>
              <p>Start one anywhere on your network:</p>
              <pre>{`from strands_robots import Robot\nRobot("so101").run()   # sim\nRobot("so101", mode="real", port="/dev/ttyACM0").run()`}</pre>
              <p className="hint">
                Set <code>STRANDS_MESH_LOCAL_DEV=1</code> + <code>STRANDS_MESH_MULTICAST=true</code> for local dev.
              </p>
              <button className="btn ghost" onClick={() => setPanel('devices')}>spawn one here</button>
            </>
          )}
        </div>
      ) : (
        <main className="grid">
          {list.map(p => (
            <RobotCard
              key={p.peer_id}
              peer={p}
              onOpen={setDetail}
              onBusyChange={(id, running) => setBusyPeers(s => (s[id] === running ? s : { ...s, [id]: running }))}
            />
          ))}
        </main>
      )}

      {detailPeer && <RobotDetail peer={detailPeer} onClose={() => setDetail(null)} />}

      <SettingsDrawer open={panel === 'settings'} onClose={() => setPanel(null)} mesh={mesh} />
      <ActivityLog open={panel === 'activity'} onClose={() => setPanel(null)} live={activity} />
      <DevicePanel open={panel === 'devices'} onClose={() => setPanel(null)} />
      <EstopSheet open={panel === 'estop'} onClose={() => setPanel(null)} />
      {panel === 'training' && <TrainingTab onClose={() => setPanel(null)} />}
      {panel === 'record' && (
        <RecordPanel peers={list.filter(p => !p.stale)} onClose={() => setPanel(null)} />
      )}

      <AgentDock
        onSettings={() => setPanel('settings')}
        startOpen={new URLSearchParams(location.search).get('panel') === 'chat'}
      />
    </div>
  )
}

/** Minimal unlock form for the 1008 case - Settings itself is behind the token. */
function TokenPrompt() {
  const [token, setToken] = useState('')
  return (
    <form
      className="tokenprompt"
      onSubmit={e => { e.preventDefault(); setAuthToken(token); location.reload() }}
    >
      <input type="password" value={token} placeholder="dashboard token"
             onChange={e => setToken(e.target.value)} />
      <button className="btn go" type="submit" disabled={!token.trim()}>unlock</button>
    </form>
  )
}

export default function App() {
  // Remount everything when the backend or token changes: sockets, peer maps and
  // frame buffers all belong to one backend. AuthGate sits INSIDE the key so a
  // freshly minted session token re-runs its open/gate probe too.
  return (
    <ConfigProvider key={backendKey()}>
      <AuthGate>
        <Dashboard />
      </AuthGate>
    </ConfigProvider>
  )
}
