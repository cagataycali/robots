import { useEffect, useState } from 'react'
import { api, post } from '../lib/endpoints'
import { CamRow, applySummary, configFromRows, rowsFromConfig } from '../lib/cameraConfig'

interface Detected { index: number; label?: string | null; in_use_by?: string | null }

/**
 * U19: per-camera reconfigure for one managed robot.
 *
 * Cameras are taken only at spawn, so applying is honestly a RESPAWN — the
 * confirm step names that cost (streams and any running task stop) instead of
 * burying it. The editor pre-fills from the child's actual spawn config
 * (/api/devices managed[peer].cameras); a peer this dashboard did not spawn
 * cannot be edited here and the sheet says so rather than offering a form
 * that can only fail.
 */
export default function CameraConfigSheet({ peerId, onClose }: { peerId: string; onClose: () => void }) {
  const [rows, setRows] = useState<CamRow[] | null>(null)
  const [detected, setDetected] = useState<Detected[]>([])
  const [notManaged, setNotManaged] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api<any>('/api/devices').then(doc => {
      if (!alive) return
      const m = doc?.managed?.[peerId]
      if (!m) { setNotManaged(true); setRows([]) }
      else setRows(rowsFromConfig(m.cameras ?? {}))
      setDetected((doc?.cameras ?? []).map((c: any) => ({
        index: c.index, label: c.label ?? c.name ?? null, in_use_by: c.claimed_by ?? null,
      })))
    }).catch(e => { if (alive) { setError(e?.message ?? String(e)); setRows([]) } })
    return () => { alive = false }
  }, [peerId])

  const edit = (i: number, patch: Partial<CamRow>) =>
    setRows(rs => rs!.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const remove = (i: number) => setRows(rs => rs!.filter((_, j) => j !== i))
  const add = (indexOrPath = '') =>
    setRows(rs => [...(rs ?? []), { name: '', indexOrPath, fps: '', width: '', height: '' }])

  const check = rows ? configFromRows(rows) : { cameras: null }

  const apply = async () => {
    if (!rows) return
    setBusy(true); setError(null)
    try {
      const r = await post<any>(`/api/devices/${encodeURIComponent(peerId)}/cameras`, { cameras: check.cameras })
      if (r?.error) setError(r.error)
      // The settle rail answers running / starting / failed / gone. `failed`
      // arrives as r.error above; `gone` means the peer despawned while we
      // watched — "watch its card come back" would be a promise about a card
      // that is never coming back, so it is an error, not a ✓.
      else if (r?.status === 'gone') setError(`${peerId} despawned during the respawn window — check devices › logs`)
      else setDone(r?.status === 'running'
        ? `✓ ${peerId} is back on the mesh with the new cameras`
        : `respawned, not yet announced on the mesh (status: ${r?.status ?? 'starting'}) — watch its card, or check devices › logs if it stays away`)
    } catch (e: any) {
      // A 404 here has one honest meaning on this codebase: the running
      // dashboard process predates this route.
      setError(e?.status === 404 && /Not Found|no such|unknown managed/i.test(e?.message ?? '')
        ? (e?.message?.includes('unknown managed')
          ? e.message
          : 'this dashboard process predates the camera-reconfigure rail — it needs a (terminal-started) restart to pick it up')
        : e?.message ?? String(e))
    } finally {
      setBusy(false); setConfirming(false)
    }
  }

  const free = detected.filter(d => !d.in_use_by || d.in_use_by === peerId)

  return (
    <div className="sheet-backdrop" onClick={busy ? undefined : onClose}>
      <div className="sheet" onClick={e => e.stopPropagation()}>
        <h3>cameras — {peerId}</h3>
        {rows === null && <p className="hint">reading the current config…</p>}
        {notManaged && (
          <p className="hint">
            This robot was not spawned by this dashboard, so its process (and its cameras)
            belong to whoever started it — change the config where it runs.
          </p>
        )}
        {rows !== null && !notManaged && !done && (
          <>
            {rows.length === 0 && <p className="hint">No cameras attached. Add one below.</p>}
            {rows.map((r, i) => (
              <div className="cam-config-row" key={i}>
                <input placeholder="name (top / wrist)" value={r.name} onChange={e => edit(i, { name: e.target.value })} />
                <input placeholder="index or path" value={r.indexOrPath} onChange={e => edit(i, { indexOrPath: e.target.value })} />
                <input placeholder="fps" inputMode="numeric" value={r.fps} onChange={e => edit(i, { fps: e.target.value })} />
                <input placeholder="width" inputMode="numeric" value={r.width} onChange={e => edit(i, { width: e.target.value })} />
                <input placeholder="height" inputMode="numeric" value={r.height} onChange={e => edit(i, { height: e.target.value })} />
                <button className="btn ghost" title="detach this camera" onClick={() => remove(i)}>detach</button>
              </div>
            ))}
            <div className="cam-config-add">
              <button className="btn ghost" onClick={() => add()}>+ add camera</button>
              {free.map(d => (
                <button key={d.index} className="btn ghost" title={d.label ?? undefined}
                        onClick={() => add(String(d.index))}>
                  + #{d.index}{d.label ? ` ${d.label}` : ''}
                </button>
              ))}
            </div>
            <p className="hint">
              Blank fps/size = the driver's defaults (640×480 @ 30). Detaching all cameras is allowed —
              the robot streams joints only.
            </p>
            {check.error && <div className="result bad">✗ {check.error}</div>}
            {error && <div className="result bad">⚠ {error}</div>}
            {!confirming ? (
              <div className="sheet-actions">
                <button className="btn go" disabled={!!check.error || busy} onClick={() => setConfirming(true)}>
                  apply…
                </button>
                <button className="btn ghost" onClick={onClose} disabled={busy}>cancel</button>
              </div>
            ) : (
              <>
                <p className="hint">{applySummary(rows, peerId)}</p>
                <div className="sheet-actions">
                  <button className="btn danger" onClick={apply} disabled={busy}>
                    {busy ? 'restarting…' : 'restart with these cameras'}
                  </button>
                  <button className="btn ghost" onClick={() => setConfirming(false)} disabled={busy}>back</button>
                </div>
              </>
            )}
          </>
        )}
        {done && (
          <>
            <div className="result ok">{done}</div>
            <div className="sheet-actions">
              <button className="btn ghost" onClick={onClose}>close</button>
            </div>
          </>
        )}
        {notManaged && (
          <div className="sheet-actions">
            <button className="btn ghost" onClick={onClose}>close</button>
          </div>
        )}
      </div>
    </div>
  )
}
