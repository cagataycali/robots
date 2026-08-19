import { useCallback, useEffect, useRef, useState } from 'react'
import { findConsent, type ConsentNeed } from '../lib/consent'
import ConsentSheet from './ConsentSheet'
import { api, post } from '../lib/endpoints'
import CalibrationSection from './CalibrationSection'
import CameraGallery, { type CameraInfo, type CameraName, type CameraProblem } from './CameraGallery'
import { normalizeRegistry, type RegistryRobot } from '../lib/registry'

interface SerialPort {
  device: string
  description?: string
  vid?: string | null
  pid?: string | null
  serial_number?: string | null
  likely_robot?: string | null
  // Measured off the servo bus (12V = follower, 7.4V = leader) and remembered by
  // serial. ABSENT means never measured, which is not the same as "unknown".
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
  role_measured_at?: number | null
}

interface RoleVerdict {
  role: string
  reason: string
  volts?: number | null
  remedy?: string
  remembered?: boolean
  remember_problem?: string
  mismatch?: { labelled: string, measured: string, message: string, remedy: string } | null
}

interface DeviceDoc {
  serial_ports: SerialPort[]
  cameras: CameraInfo[]
  camera_names?: CameraName[]
  camera_problem?: CameraProblem | null
  managed: Record<string, Managed>
}

interface Managed {
  peer_id: string
  robot_name: string
  mode: string
  port?: string | null
  alive: boolean
  started_at?: number
  log_tail?: string[]
}

/**
 * Local hardware, and the robot processes this dashboard owns.
 *
 * A managed robot is a *child process* that joins the mesh as its own peer, so
 * it appears twice: as a peer card in the fleet and as a row here. This is the
 * only place that can kill it, and the only place its stdout is visible —
 * a robot that dies during `Robot()` construction never reaches the mesh at all,
 * and its traceback is in that log tail.
 */
export default function DevicePanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [doc, setDoc] = useState<DeviceDoc | null>(null)
  const [robots, setRobots] = useState<RegistryRobot[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [logs, setLogs] = useState<{ peer: string; lines: string[] } | null>(null)
  // A spawn refused by a safety guard (U18): keep the request so approving can
  // re-run exactly what was refused.
  const [consent, setConsent] = useState<ConsentNeed | null>(null)
  const retry = useRef<{ fn: () => Promise<any>; label: string } | null>(null)

  // spawn form
  const [robotName, setRobotName] = useState('')
  const [mode, setMode] = useState<'sim' | 'real'>('sim')
  const [port, setPort] = useState('')
  const [camIndex, setCamIndex] = useState('')
  const [camFps, setCamFps] = useState('')
  const [camW, setCamW] = useState('')
  const [camH, setCamH] = useState('')
  const [robotId, setRobotId] = useState('')
  // Measured servo-bus roles, keyed by port. Declared here with the other hooks
  // on purpose: this component returns null when closed, so a useState below
  // that bail-out changes the hook count between renders (React #310).
  const [roles, setRoles] = useState<Record<string, RoleVerdict>>({})
  const [measuring, setMeasuring] = useState<string | null>(null)

  const load = useCallback(async (refresh = false) => {
    try {
      setDoc(await api<DeviceDoc>(`/api/devices${refresh ? '?refresh=1' : ''}`))
      setError(null)
    } catch (e: any) {
      setError(e?.message ?? String(e))
    }
  }, [])

  useEffect(() => {
    if (!open) return
    void load()
    void api<{ robots: any }>('/api/robots/registry')
      .then(r => setRobots(normalizeRegistry(r.robots)))
      .catch(() => setRobots([]))
    // Managed children die on their own (import errors, unplugged bus); poll so
    // an `alive: false` row does not sit there looking healthy.
    const id = setInterval(() => void load(), 5000)
    return () => clearInterval(id)
  }, [open, load])

  if (!open) return null

  const act = async (fn: () => Promise<any>, label: string) => {
    setBusy(true); setStatus(null); setConsent(null)
    retry.current = { fn, label }
    try {
      const r = await fn()
      setStatus(r?.error ? `⚠ ${r.error}` : `${label}: ${r?.peer_id ?? 'ok'}`)
      // 200-with-error is how a settled-then-dead spawn reports itself; the
      // consent hint rides in that same body.
      setConsent(findConsent(r))
      await load()
    } catch (e: any) {
      setStatus(`⚠ ${e?.message ?? String(e)}`)
      setConsent(findConsent(e?.body))
    } finally {
      setBusy(false)
    }
  }

  const spawn = () => act(() => post('/api/devices/spawn', {
    robot_name: robotName,
    mode,
    port: mode === 'real' ? port || null : null,
    // The camera config must be a MAPPING per entry ({index_or_path: N, ...});
    // a bare int here is the exact ValueError an operator once hit live:
    // "Camera 'main' config must be a mapping ... got int: 3".
    cameras: camIndex === '' ? null : {
      main: {
        index_or_path: Number(camIndex),
        ...(camFps !== '' ? { fps: Number(camFps) } : {}),
        ...(camW !== '' ? { width: Number(camW) } : {}),
        ...(camH !== '' ? { height: Number(camH) } : {}),
      },
    },
    robot_id: robotId || null,
  }), 'spawned')

  const showLogs = async (peer: string) => {
    try {
      const r = await api<{ lines: string[] }>(`/api/devices/logs/${encodeURIComponent(peer)}`)
      setLogs({ peer, lines: r.lines ?? [] })
    } catch (e: any) {
      setLogs({ peer, lines: [`⚠ ${e?.message ?? String(e)}`] })
    }
  }

  const measureRole = async (port: string) => {
    setMeasuring(port)
    try {
      const v = await api<RoleVerdict>(`/api/devices/arm-role?port=${encodeURIComponent(port)}`)
      setRoles(r => ({ ...r, [port]: v }))
      // The verdict is remembered server-side against the board's serial, so
      // reload to pick up the badge rather than mirroring it in two places.
      if (v.remembered) void load(false)
    } catch (e: any) {
      setRoles(r => ({
        ...r,
        [port]: { role: 'unknown', reason: e?.message ?? String(e) },
      }))
    } finally {
      setMeasuring(null)
    }
  }

  const managed = Object.values(doc?.managed ?? {})
  const freePorts = doc?.serial_ports ?? []
  const claimedPorts = new Set(managed.filter(m => m.alive && m.port).map(m => m.port as string))

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer wide" onClick={e => e.stopPropagation()}>
        <header className="drawer-head">
          <h2>Devices</h2>
          <div>
            <button className="btn ghost" onClick={() => void load(true)} disabled={busy}>rescan</button>
            <button className="btn ghost" onClick={onClose}>✕</button>
          </div>
        </header>

        <div className="drawer-body">
          {error && <div className="result bad">⚠ {error}</div>}
          {consent && (
            <ConsentSheet
              need={consent}
              target="spawn"
              onCancel={() => setConsent(null)}
              onRetry={() => { const again = retry.current; setConsent(null); if (again) void act(again.fn, again.label) }}
            />
          )}

          <section>
            <h3>Managed robots ({managed.length})</h3>
            {managed.length === 0 && (
              <p className="hint">
                None. Spawning one starts a child process that joins the mesh as its own peer —
                use it for a MuJoCo sim, or to drive a real arm from this machine.
              </p>
            )}
            <ul className="devlist">
              {managed.map(m => (
                <li key={m.peer_id} className={m.alive ? '' : 'dead'}>
                  <span className={m.alive ? 'dot on' : 'dot off'} />
                  <b>{m.peer_id}</b>
                  <span className="meta">
                    {m.robot_name} · {m.mode}{m.port ? ` · ${m.port}` : ''}
                    {!m.alive && ' · exited'}
                  </span>
                  <span className="devactions">
                    <button className="btn ghost" onClick={() => void showLogs(m.peer_id)}>logs</button>
                    <button className="btn ghost danger" disabled={busy}
                            onClick={() => void act(() => post('/api/devices/despawn', { peer_id: m.peer_id }), 'despawned')}>
                      {m.alive ? 'despawn' : 'remove'}
                    </button>
                  </span>
                  {!m.alive && m.log_tail?.length ? (
                    <pre className="logtail">{m.log_tail.slice(-6).join('\n')}</pre>
                  ) : null}
                </li>
              ))}
            </ul>
          </section>

          {logs && (
            <section>
              <h3>
                {logs.peer} log
                <button className="btn ghost" onClick={() => setLogs(null)}>close</button>
              </h3>
              <pre className="logtail tall">{logs.lines.join('\n') || '(no output yet)'}</pre>
            </section>
          )}

          <section>
            <h3>Spawn</h3>
            <div className="row">
              <label className="field">
                <span>Robot</span>
                <select value={robotName} onChange={e => setRobotName(e.target.value)}>
                  <option value="">select…</option>
                  {robots.map(r => <option key={r.name} value={r.name}>{r.label}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Mode</span>
                <select value={mode} onChange={e => setMode(e.target.value as 'sim' | 'real')}>
                  <option value="sim">sim (MuJoCo)</option>
                  <option value="real">real hardware</option>
                </select>
              </label>
            </div>
            {mode === 'real' && (
              <>
                <div className="row">
                  <label className="field">
                    <span>Servo bus</span>
                    <select value={port} onChange={e => setPort(e.target.value)}>
                      <option value="">select a port…</option>
                      {freePorts.map(p => (
                        <option key={p.device} value={p.device} disabled={claimedPorts.has(p.device)}>
                          {p.device}{p.likely_robot ? ` (${p.likely_robot})` : ''}
                          {claimedPorts.has(p.device) ? ' — in use' : ''}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Calibration id</span>
                    <input value={robotId} placeholder="lerobot id (optional)"
                           onChange={e => setRobotId(e.target.value)} />
                  </label>
                </div>
                <p className="hint">
                  A real robot moves as soon as a task runs. The calibration id must match the one
                  used by <code>lerobot-calibrate</code>, or the joint limits will be wrong.
                </p>
              </>
            )}
            <label className="field">
              <span>Camera</span>
              <select value={camIndex} onChange={e => setCamIndex(e.target.value)}>
                <option value="">none</option>
                {(doc?.cameras ?? []).map(c => (
                  <option key={c.index} value={c.index} disabled={!!c.claimed_by}>
                    index {c.index}
                    {c.width ? ` — ${c.width}×${c.height}` : ''}
                    {c.fps ? ` @ ${c.fps}fps` : ''}
                    {c.claimed_by ? ` — claimed by ${c.claimed_by}` : ''}
                  </option>
                ))}
              </select>
            </label>
            {camIndex !== '' && (
              <>
                <div className="row">
                  <label className="field">
                    <span>FPS</span>
                    <input type="number" inputMode="numeric" min={1} max={120}
                           value={camFps} placeholder="30"
                           onChange={e => setCamFps(e.target.value)} />
                  </label>
                  <label className="field">
                    <span>Width</span>
                    <input type="number" inputMode="numeric" min={64} step={2}
                           value={camW} placeholder="640"
                           onChange={e => setCamW(e.target.value)} />
                  </label>
                  <label className="field">
                    <span>Height</span>
                    <input type="number" inputMode="numeric" min={64} step={2}
                           value={camH} placeholder="480"
                           onChange={e => setCamH(e.target.value)} />
                  </label>
                </div>
                <p className="hint">
                  Blank = the driver's defaults (640×480 @ 30). A setting the camera can't do
                  fails loudly at spawn — check the log tail, not the stream.
                </p>
              </>
            )}
            <div className="sheet-actions">
              <button className="btn go" disabled={busy || !robotName || (mode === 'real' && !port)}
                      onClick={spawn}>
                spawn
              </button>
            </div>
          </section>

          <section>
            <h3>Cameras</h3>
            <CameraGallery cameras={doc?.cameras ?? []} names={doc?.camera_names ?? []}
                           problem={doc?.camera_problem ?? null} />
            <p className="hint">
              Camera indices owned by a running robot are never re-probed — opening one steals
              frames from its capture thread mid-episode.
            </p>
          </section>

          <CalibrationSection />

          <section>
            <h3>Servo boards</h3>
            <p className="hint">
              A follower arm runs a 12V servo bus, a leader 7.4V — so the role can be read off the
              hardware instead of inherited from a name. The read touches one register
              (<code>Present_Voltage</code>) and cannot move the arm. A servo bus has a single owner,
              so an arm that is running must be despawned before it can be measured.
            </p>
            <ul className="boardlist">
              {freePorts.length === 0 && <li className="muted">no servo board detected</li>}
              {freePorts.map(p => {
                const v = roles[p.device]
                const busy = claimedPorts.has(p.device)
                return (
                  <li key={p.device}>
                    <div className="row between">
                      <span className="mono">{p.device}</span>
                      {p.role
                        ? <span className={'rolebadge ' + p.role}>
                            {p.role}{p.role_volts ? ' · ' + p.role_volts + 'V' : ''}
                          </span>
                        : <span className="rolebadge unmeasured">role not measured</span>}
                    </div>
                    <div className="row between">
                      <span className="muted small">
                        {p.serial_number ? 'serial ' + p.serial_number : 'no serial number'}
                      </span>
                      <button className="btn ghost" disabled={busy || measuring === p.device}
                              title={busy
                                ? 'this arm is running and owns its bus — despawn it first'
                                : 'reads one register; cannot move the arm'}
                              onClick={() => void measureRole(p.device)}>
                        {measuring === p.device
                          ? 'reading…'
                          : busy ? 'running — despawn to measure' : 'measure role'}
                      </button>
                    </div>
                    {v && (
                      <p className={v.mismatch ? 'warn small' : 'muted small'}>
                        {v.reason}
                        {v.remedy ? ' — ' + v.remedy : ''}
                        {v.mismatch ? ' ⚠ ' + v.mismatch.message + ' — ' + v.mismatch.remedy : ''}
                        {v.remembered ? ' · remembered for this board' : ''}
                        {v.remember_problem ? ' · ' + v.remember_problem : ''}
                      </p>
                    )}
                  </li>
                )
              })}
            </ul>
          </section>

          <section>
            <h3>Detected hardware</h3>
            <dl className="kv">
              <dt>serial</dt>
              <dd className="mono">
                {freePorts.length
                  ? freePorts.map(p => p.device).join(', ')
                  : 'none (a servo bus shows up as /dev/tty.usbmodem* or /dev/ttyACM*)'}
              </dd>
              <dt>cameras</dt>
              <dd className="mono">
                {doc?.cameras.length
                  ? doc.cameras.map(c => `#${c.index}${c.claimed_by ? `→${c.claimed_by}` : ''}`).join(', ')
                  : 'none probed'}
              </dd>
            </dl>
          </section>
        </div>

        {status && <footer className="drawer-foot">{status}</footer>}
      </aside>
    </div>
  )
}
