import { useMesh } from './lib/useMesh'
import FleetBar from './components/FleetBar'
import RobotCard from './components/RobotCard'

export default function App() {
  const { conn, dashboardId, peers, safetyFlash } = useMesh()
  const list = Object.values(peers).sort((a, b) => {
    // fresh first, then robots before sims, then name
    if (!!a.stale !== !!b.stale) return a.stale ? 1 : -1
    const at = a.presence?.robot_type ?? 'z', bt = b.presence?.robot_type ?? 'z'
    if (at !== bt) return at === 'robot' ? -1 : 1
    return a.peer_id.localeCompare(b.peer_id)
  })

  return (
    <div className="stage">
      <FleetBar conn={conn} peerCount={list.filter(p => !p.stale).length} dashboardId={dashboardId} safetyFlash={safetyFlash} />

      {list.length === 0 ? (
        <div className="empty-fleet">
          <div className="empty-icon">📡</div>
          <h2>No robots on the mesh yet</h2>
          <p>Start one anywhere on your network:</p>
          <pre>{`from strands_robots import Robot\nRobot("so101").run()   # sim\nRobot("so101", mode="real", port="/dev/ttyACM0").run()`}</pre>
          <p className="hint">Set <code>STRANDS_MESH_LOCAL_DEV=1</code> + <code>STRANDS_MESH_MULTICAST=true</code> for local dev.</p>
        </div>
      ) : (
        <main className="grid">
          {list.map(p => <RobotCard key={p.peer_id} peer={p} />)}
        </main>
      )}
    </div>
  )
}
