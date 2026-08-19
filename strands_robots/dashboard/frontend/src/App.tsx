import { useEffect, useMemo, useState } from 'react'
import { useMesh } from './lib/useMesh'
import { usePwa } from './lib/usePwa'
import { linkHealth, estopPosture } from './lib/linkHealth'
import { ConfigProvider } from './lib/useConfig'
import { backendKey, backendLabel, setAuthToken } from './lib/endpoints'
import FleetBar from './components/FleetBar'
import { getRecordApi } from './lib/recordApi'
import RobotCard from './components/RobotCard'
import RobotDetail from './components/RobotDetail'
import AgentDock from './components/AgentDock'
import SettingsDrawer from './components/SettingsDrawer'
import ActivityLog from './components/ActivityLog'
import DevicePanel from './components/DevicePanel'
import EstopSheet from './components/EstopSheet'
import HelpSheet from './components/HelpSheet'
import EstopButton from './components/EstopButton'
import { hotkeyVerdict } from './lib/hotkeys'
import ErrorBoundary from './components/ErrorBoundary'
import TrainingTab from './components/TrainingTab'
import RecordPanel from './components/RecordPanel'
import AuthGate from './components/AuthGate'

type Panel = 'settings' | 'activity' | 'devices' | 'estop' | 'training' | 'record' | 'help' | null

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
  const { conn, dashboardId, peers, safetyFlash, mesh, activity, loaded, lastEventAt, everOpen } = useMesh()
  const pwa = usePwa()
  const [panel, setPanel] = useState<Panel>(initialPanel)
  const [settingsTab, setSettingsTab] = useState<'mesh' | undefined>(undefined)
  const [detail, setDetail] = useState<string | null>(null)
  const [busyPeers, setBusyPeers] = useState<Record<string, boolean>>({})
  /* UX_REVIEW #10: the record backend is probed ONCE per page load (lib/recordApi
     caches it), so asking here costs nothing and lets the nav warn before the
     click instead of after. null until the answer arrives — the nav must not
     guess in either direction. */
  const [recordMock, setRecordMock] = useState<boolean | null>(null)

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
  useEffect(() => {
    let live = true
    void getRecordApi().then(a => { if (live) setRecordMock(a.mock) }).catch(() => {})
    return () => { live = false }
  }, [])

  // Keyboard: Escape closes, "." (or Cmd/Ctrl+. even while typing) opens the
  // stop sheet, "?" opens help. The decision itself is pure and tested in
  // lib/hotkeys.ts — including the case that made JOURNEYS #12 dangerous: an
  // operator mid-sentence in a form, where the bare key must stay a character
  // and the brake still has to be one chord away.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      const verdict = hotkeyVerdict({
        key: e.key, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey, shiftKey: e.shiftKey,
        targetTag: el?.tagName, editable: el?.isContentEditable, repeat: e.repeat,
      })
      if (!verdict) return
      if (verdict === 'close') { setPanel(null); return }
      // The chord must not also reach the browser (Cmd+. is "stop loading" in
      // some builds) or insert anything into the field it was pressed in.
      if (e.metaKey || e.ctrlKey) e.preventDefault()
      setPanel(verdict === 'estop' ? 'estop' : 'help')
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

  // Re-evaluated on a 1s tick ONLY while something is wrong, so the "frozen
  // (Ns old)" number keeps counting instead of freezing with the view it
  // describes. A healthy link does no work here.
  const [linkTick, setLinkTick] = useState(0)
  const link = linkHealth({
    conn, browserOnline: pwa.online, lastEventAt, everOpen,
    // list.length, NOT the non-stale count: what is RENDERED is what can
    // mislead. Measured — when the stream dies quietly the stale sweep drops
    // the non-stale count to 0, so gating on it went silent at the exact moment
    // the screen was worst (2 frozen cards, no banner, a normal-looking brake).
    peerCount: list.length, now: Date.now(),
  })
  // The tick runs ALWAYS, and that is the whole point. I first gated it on the
  // verdict being unhealthy and MEASURED the result: with the link healthy
  // nothing schedules a render, so when the stream went quiet the component
  // never re-rendered, never recomputed the verdict, and sat there showing a
  // live-looking fleet with a normal brake — until an unrelated click forced a
  // render 23 s later. A watchdog gated on its own subject cannot fire. 1 Hz is
  // free next to the telemetry renders a healthy fleet already causes.
  useEffect(() => {
    const id = setInterval(() => setLinkTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])
  void linkTick

  return (
    <div className="stage">
      {/* FIRST IN THE DOM, therefore the FIRST TAB STOP on every screen
          (JOURNEYS #12: measured 14 to 30+ tab stops to reach it, and on the
          training screen it was unreachable inside 30). It is position:fixed in
          its own layer, so moving it to the top of the document changes nothing
          visually and no overlay can swallow the stop. A keyboard user's brake
          must not be behind the fleet bar's chips, a robot card's controls, or
          whatever a drawer happens to render today. */}
      <EstopButton onClick={() => setPanel('estop')} posture={estopPosture(link)} />

      <FleetBar
        conn={conn}
        peerCount={list.filter(p => !p.stale).length}
        dashboardId={dashboardId}
        safetyFlash={safetyFlash}
        mesh={mesh}
        online={pwa.online}
        installable={pwa.installable}
        activityCount={activity.length}
        recordMock={recordMock}
        onInstall={() => void pwa.install()}
        onSettings={() => { setSettingsTab(undefined); setPanel('settings') }}
        onWireSecurity={() => { setSettingsTab('mesh'); setPanel('settings') }}
        onActivity={() => setPanel('activity')}
        onDevices={() => setPanel('devices')}
        onTraining={() => setPanel('training')}
        onRecord={() => setPanel('record')}
        onHelp={() => setPanel('help')}
      />


      {pwa.needRefresh && (
        <div className="toast">
          A new dashboard version is ready.
          <button className="btn go" onClick={pwa.update}>reload</button>
          <span className="hint">Reloading drops camera streams; a running task keeps running.</span>
        </div>
      )}

      {/* One judgment for all the ways this page can stop being attached to the
          fleet (lib/linkHealth): this device's network, a refused token, a dead
          API, a mute socket. The old toast covered only the first — yet the
          measured outage was the API dying twice in 30 minutes, which showed a
          frozen fleet and a brake that looked fine. */}
      {link.headline && (
        <div className={`toast ${link.commandsWork ? '' : 'warn'}`} role="status">
          <b>{link.headline}</b> {link.detail}
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
            <ErrorBoundary key={p.peer_id} label={`the card for ${p.peer_id}`}>
            <RobotCard
              key={p.peer_id}
              peer={p}
              onOpen={setDetail}
              onBusyChange={(id, running) => setBusyPeers(s => (s[id] === running ? s : { ...s, [id]: running }))}
            />
            </ErrorBoundary>
          ))}
        </main>
      )}

      {detailPeer && (
        <ErrorBoundary label="the robot detail view" onDismiss={() => setDetail(null)}>
          <RobotDetail peer={detailPeer} onClose={() => setDetail(null)} />
        </ErrorBoundary>
      )}

      <ErrorBoundary label="settings" onDismiss={() => setPanel(null)}>
        <SettingsDrawer open={panel === 'settings'} onClose={() => setPanel(null)} mesh={mesh} initialTab={settingsTab} />
      </ErrorBoundary>
      <ErrorBoundary label="the activity log" onDismiss={() => setPanel(null)}>
        <ActivityLog open={panel === 'activity'} onClose={() => setPanel(null)} live={activity} />
      </ErrorBoundary>
      <ErrorBoundary label="the devices screen" onDismiss={() => setPanel(null)}>
        <DevicePanel open={panel === 'devices'} onClose={() => setPanel(null)} />
      </ErrorBoundary>
      <HelpSheet open={panel === 'help'} onClose={() => setPanel(null)} />
      <EstopSheet open={panel === 'estop'} onClose={() => setPanel(null)}
        linkWarning={link.commandsWork ? null : link.estopReason} />
      {panel === 'training' && (
        <ErrorBoundary label="the training screen" onDismiss={() => setPanel(null)}>
          <TrainingTab onClose={() => setPanel(null)} />
        </ErrorBoundary>
      )}
      {panel === 'record' && (
        <ErrorBoundary label="the record screen" onDismiss={() => setPanel(null)}>
          <RecordPanel peers={list.filter(p => !p.stale)} onClose={() => setPanel(null)} />
        </ErrorBoundary>
      )}

      <ErrorBoundary label="the chat dock">
        <AgentDock
        onSettings={() => setPanel('settings')}
        startOpen={new URLSearchParams(location.search).get('panel') === 'chat'}
        exampleRobot={list.find(p => !p.stale && p.presence?.robot_type === 'robot')?.peer_id}
        />
      </ErrorBoundary>
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
