import { useEffect, useState } from 'react'
import type { Peer } from '../types'
import { getRecordApi, type RecordApi, type RecordSession } from '../lib/recordApi'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'

/**
 * Record screen (U8): collect teleop episodes into a LeRobotDataset.
 * The leader arm drives the follower; the follower is what gets recorded.
 *
 * Session state lives on the backend (see lib/recordApi.ts) - this component
 * only renders it and sends intents, so a phone and a laptop can watch the
 * same session. While /api/record is unbuilt the api is an honest mock, and
 * the banner says so.
 */
export default function RecordPanel({ peers, onClose }: { peers: Peer[]; onClose: () => void }) {
  const peerIds = peers.map(p => p.peer_id)
  const [api, setApi] = useState<RecordApi | null>(null)
  const [s, setS] = useState<RecordSession | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const guessLeader = peerIds.find(p => /leader|arm-2/.test(p)) ?? peerIds[1] ?? ''
  const guessFollower = peerIds.find(p => /follower|arm-1/.test(p)) ?? peerIds[0] ?? ''
  const [form, setForm] = useState({
    dataset: '', task: '', leader: guessLeader, follower: guessFollower, target_episodes: '20',
  })
  const [upload, setUpload] = useState(false)
  const [repoId, setRepoId] = useState('')
  const [closed, setClosed] = useState<string | null>(null)

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

      {open && s && <FollowerLive peer={peers.find(p => p.peer_id === s.follower)} recording={recording} />}

      {open && s && s.episodes.length > 0 && (
        <div className="rec-strip" role="list" aria-label="recorded episodes">
          {s.episodes.slice().reverse().map(ep => (
            <div key={ep.index} role="listitem" className={`rec-ep${ep.discarded ? ' dead' : ''}`}>
              <div className="rec-ep-head">
                <b>ep {ep.index}</b>
                <span>{ep.frames}f · {ep.duration_s}s</span>
              </div>
              {Object.entries(ep.thumbnails).length > 0 && (
                <div className="rec-thumbs">
                  {Object.entries(ep.thumbnails).slice(0, 3).map(([cam, url]) => (
                    <img key={cam} src={url} alt={`${cam} thumbnail of episode ${ep.index}`} loading="lazy" />
                  ))}
                </div>
              )}
              {ep.discarded
                ? <span className="rec-ep-gone">discarded</span>
                : <button className="btn ghost" onClick={() => void run(() => api!.discard(ep.index))} disabled={busy}>
                    ✕ discard
                  </button>}
            </div>
          ))}
        </div>
      )}

      {open && s && !recording && (
        <div className="train-form">
          <label className="field check">
            <input type="checkbox" checked={upload} onChange={e => setUpload(e.target.checked)} />
            <span>upload to the Hugging Face Hub after finishing</span>
          </label>
          {upload && (
            <label className="field"><span>hub repo id <em>(defaults to the dataset name)</em></span>
              <input value={repoId} placeholder={s.dataset ?? ''} onChange={e => setRepoId(e.target.value)} />
            </label>
          )}
          <div className="train-actions">
            <button className="btn wide" disabled={busy} onClick={() => {
              void (async () => {
                setBusy(true); setErr(null)
                try {
                  const r = await api!.close(upload ? { upload, repo_id: repoId.trim() || s.dataset || undefined } : {})
                  setClosed(r.detail ?? (r.ok ? `dataset finished with ${kept} episode(s)` : 'close failed'))
                  setS(await api!.session())
                } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
                setBusy(false)
              })()
            }}>
              ✓ finish dataset ({kept} kept{s.episodes.length - kept ? `, ${s.episodes.length - kept} discarded` : ''})
            </button>
          </div>
        </div>
      )}

      {closed && !open && <div className="toast">✓ {closed}</div>}

      {err && <div className="train-msg">✗ {err}</div>}
    </div>
  )
}

/**
 * What the dataset sees: the follower's cameras and joints, live during the
 * whole session (not just while recording - the operator lines the arms up
 * BETWEEN episodes, and needs eyes then most of all).
 */
function FollowerLive({ peer, recording }: { peer?: Peer; recording: boolean }) {
  if (!peer) {
    return <div className="toast warn">The follower is not on the mesh — its card left the fleet. Recording would capture nothing.</div>
  }
  const cams = Object.keys(peer.cameras ?? {})
  return (
    <div className={`train-form rec-live-view${recording ? ' armed' : ''}`}>
      <div className="rec-live-head">
        <span>{peer.peer_id} — what the dataset sees</span>
        {recording && <span className="rec-dot" aria-hidden="true" />}
      </div>
      {cams.length > 0 ? (
        <div className={cams.length > 1 ? 'cams multi' : 'cams'}>
          {cams.slice(0, 4).map(c => (
            <CameraTile key={c} peerId={peer.peer_id} cam={c} meta={peer.cameras?.[c]} />
          ))}
        </div>
      ) : (
        <div className="train-msg">⚠ no cameras announced by this peer — the dataset will have joints only</div>
      )}
      <JointStrip state={peer.state} presence={peer.presence} />
    </div>
  )
}
