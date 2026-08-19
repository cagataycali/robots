import { useEffect, useState } from 'react'
import { getRecordApi, type RecordApi, type RecordSession } from '../lib/recordApi'

/**
 * Record screen (U8): collect teleop episodes into a LeRobotDataset.
 * The leader arm drives the follower; the follower is what gets recorded.
 *
 * Session state lives on the backend (see lib/recordApi.ts) - this component
 * only renders it and sends intents, so a phone and a laptop can watch the
 * same session. While /api/record is unbuilt the api is an honest mock, and
 * the banner says so.
 */
export default function RecordPanel({ peerIds, onClose }: { peerIds: string[]; onClose: () => void }) {
  const [api, setApi] = useState<RecordApi | null>(null)
  const [s, setS] = useState<RecordSession | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const guessLeader = peerIds.find(p => /leader|arm-2/.test(p)) ?? peerIds[1] ?? ''
  const guessFollower = peerIds.find(p => /follower|arm-1/.test(p)) ?? peerIds[0] ?? ''
  const [form, setForm] = useState({
    dataset: '', task: '', leader: guessLeader, follower: guessFollower, target_episodes: '20',
  })

  useEffect(() => {
    let alive = true
    getRecordApi().then(a => {
      if (!alive) return
      setApi(a)
      a.session().then(sess => alive && setS(sess)).catch(e => alive && setErr(String(e)))
    })
    return () => { alive = false }
  }, [])

  const run = async (fn: () => Promise<RecordSession>) => {
    if (busy) return
    setBusy(true); setErr(null)
    try { setS(await fn()) } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    setBusy(false)
  }

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))
  const kept = s?.episodes.filter(e => !e.discarded).length ?? 0
  const open = !!s?.dataset
  const recording = s?.phase === 'recording'

  return (
    <div className="train-sheet" role="dialog" aria-label="Record episodes">
      <div className="train-head">
        <h2>⏺ Record</h2>
        <button className="dock-min" onClick={onClose} aria-label="close">✕</button>
      </div>

      {api?.mock && (
        <div className="toast warn">
          The backend has no /api/record yet — this is a rehearsal. Nothing is written to disk.
        </div>
      )}

      {!open && s && (
        <form className="train-form" onSubmit={e => {
          e.preventDefault()
          void run(() => api!.open({
            dataset: form.dataset.trim(), task: form.task.trim(),
            leader: form.leader, follower: form.follower,
            target_episodes: Math.max(1, Number(form.target_episodes) || 20),
          }))
        }}>
          <label className="field"><span>dataset (name or hf repo id)</span>
            <input value={form.dataset} placeholder="cagatay/so101-pick-cube"
                   onChange={e => set('dataset', e.target.value)} />
          </label>
          <label className="field"><span>task — what the arm is being taught</span>
            <input value={form.task} placeholder="pick up the red cube and place it in the bin"
                   onChange={e => set('task', e.target.value)} />
          </label>
          <div className="train-row">
            <label className="field"><span>leader (you move this one)</span>
              <select value={form.leader} onChange={e => set('leader', e.target.value)}>
                {peerIds.map(p => <option key={p}>{p}</option>)}
              </select>
            </label>
            <label className="field"><span>follower (gets recorded)</span>
              <select value={form.follower} onChange={e => set('follower', e.target.value)}>
                {peerIds.map(p => <option key={p}>{p}</option>)}
              </select>
            </label>
            <label className="field"><span>episodes</span>
              <input inputMode="numeric" value={form.target_episodes}
                     onChange={e => set('target_episodes', e.target.value)} />
            </label>
          </div>
          <div className="train-actions">
            <button className="btn go wide" type="submit"
                    disabled={busy || !api || !form.dataset.trim() || !form.task.trim() || form.leader === form.follower}>
              open session
            </button>
          </div>
          {form.leader === form.follower && <div className="train-msg">⚠ leader and follower must be different arms</div>}
        </form>
      )}

      {open && s && (
        <div className="train-form">
          <div className="rec-counter" aria-live="polite">
            <b>{kept}</b><span> / {s.target_episodes} episodes</span>
            <span className="rec-task">{s.dataset} — “{s.task}”</span>
          </div>
          <div className="train-actions">
            {!recording ? (
              <button className="btn go wide" onClick={() => void run(() => api!.startEpisode())} disabled={busy}>
                ⏺ start episode {kept + 1}
              </button>
            ) : (
              <>
                <button className="btn wide" onClick={() => void run(() => api!.redoEpisode())} disabled={busy}>
                  ↺ redo
                </button>
                <button className="btn go wide rec-live" onClick={() => void run(() => api!.stopEpisode())} disabled={busy}>
                  ⏹ stop &amp; keep
                </button>
              </>
            )}
          </div>
          <div className="train-msg">
            {recording
              ? `recording — drive ${s.follower} with ${s.leader}, then stop (or redo to throw this one away)`
              : 'start when both arms are in position'}
          </div>
        </div>
      )}

      {err && <div className="train-msg">✗ {err}</div>}
    </div>
  )
}
