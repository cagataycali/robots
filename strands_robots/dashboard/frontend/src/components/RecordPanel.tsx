import { useEffect, useRef, useState } from 'react'
import type { Peer } from '../types'
import { getRecordApi, type RecordApi, type RecordSession } from '../lib/recordApi'
import { openActionCopy } from '../lib/recordAction'
import { api as httpGet } from '../lib/endpoints'
import { pairArms, roleLabel, contradiction, type RoleCandidate } from '../lib/armPairing'
import CameraTile from './CameraTile'
import JointStrip from './JointStrip'

/**
 * Record screen (U8): collect teleop episodes into a LeRobotDataset.
 * The leader arm drives the follower; the follower is what gets recorded.
 *
 * Session state lives on the backend (see lib/recordApi.ts) - this component
 * only renders it and sends intents, and it POLLS while a session is open, so
 * a phone and a laptop really do watch the same session (frame counts tick,
 * a stop pressed on one device appears on the other). Space starts/stops an
 * episode and X redoes a bad one - collection is a two-handed job and the
 * operator's eyes are on the arms, not the pointer.
 */
export default function RecordPanel({ peers, onClose }: { peers: Peer[]; onClose: () => void }) {
  const peerIds = peers.map(p => p.peer_id)
  const [api, setApi] = useState<RecordApi | null>(null)
  const [s, setS] = useState<RecordSession | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  // Which arm is which comes from the servo bus, not from the peer id: the old
  // default here was /leader|arm-2/, and arm-2 measures 12.6V - it is the
  // FOLLOWER - so this screen used to pre-fill the pair backwards. Roles arrive
  // from /api/devices (measured once, remembered by USB serial). Until they do,
  // both slots stay empty rather than confidently wrong.
  const [roles, setRoles] = useState<Record<string, RoleCandidate>>({})
  const candidates: RoleCandidate[] = peerIds.map(id => roles[id] ?? { peer_id: id })
  const suggestion = pairArms(candidates)
  const [form, setForm] = useState({
    dataset: '', task: '', leader: '', follower: '', target_episodes: '20',
  })
  const [touched, setTouched] = useState({ leader: false, follower: false })
  // A pair the hardware contradicts is not forbidden - a bench rig can be wired
  // in a way we cannot see - but it is not a silent default either. This follows
  // the same posture as the other safety refusals in this dashboard: state the
  // measurement, then let the operator continue DELIBERATELY.
  const [ack, setAck] = useState(false)
  // Every slot the measurement contradicts, in one place: the warnings, the
  // acknowledgement and the submit gate must agree about what is wrong.
  const problems = (['leader', 'follower'] as const)
    .map(slot => ({ slot, msg: contradiction(candidates, slot, form[slot]) }))
    .filter((x): x is { slot: 'leader' | 'follower'; msg: string } => !!x.msg)

  const [upload, setUpload] = useState(false)
  const [repoId, setRepoId] = useState('')
  const [closed, setClosed] = useState<string | null>(null)

  // The measured roles. Managed children carry them keyed by peer id, so this is
  // one request, not one per arm. A failure here is not worth a banner: the
  // pickers simply stay unopinionated (basis 'none' says so out loud).
  useEffect(() => {
    let alive = true
    httpGet<{ managed?: Record<string, RoleCandidate & { peer_id: string }> }>('/api/devices')
      .then(doc => {
        if (!alive) return
        const next: Record<string, RoleCandidate> = {}
        for (const m of Object.values(doc.managed ?? {})) {
          if (m?.peer_id) next[m.peer_id] = { peer_id: m.peer_id, role: m.role, role_volts: m.role_volts }
        }
        setRoles(next)
      })
      .catch(() => { /* unmeasured is a valid state, not an error */ })
    return () => { alive = false }
  }, [])

  // Fill a slot the operator has not touched. Their choice always wins - a late
  // arriving measurement must never move a selection under their hands.
  useEffect(() => {
    setForm(f => ({
      ...f,
      leader: touched.leader ? f.leader : suggestion.leader,
      follower: touched.follower ? f.follower : suggestion.follower,
    }))
  }, [suggestion.leader, suggestion.follower, touched.leader, touched.follower])

  useEffect(() => {
    let alive = true
    getRecordApi().then(a => {
      if (!alive) return
      setApi(a)
      a.session().then(sess => alive && setS(sess)).catch(e => alive && setErr(String(e)))
    })
    return () => { alive = false }
  }, [])

  // Poll while a session is open: frame counts tick during an episode and a
  // second device (the phone next to the arms) stays in sync. One request in
  // flight at a time; the mock answers from memory so polling it is free.
  const sRef = useRef(s); sRef.current = s
  useEffect(() => {
    if (!api || !s?.dataset) return
    let alive = true
    let pending = false
    const t = setInterval(() => {
      if (pending) return
      pending = true
      api.session()
        .then(sess => { if (alive) setS(sess) })
        .catch(() => { /* a blip; the next tick retries */ })
        .finally(() => { pending = false })
    }, 1000)
    return () => { alive = false; clearInterval(t) }
  }, [api, s?.dataset])

  const run = async (fn: () => Promise<RecordSession>) => {
    if (busy) return
    setBusy(true); setErr(null)
    try { setS(await fn()) } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    setBusy(false)
  }

  // Collection is a two-handed job - eyes on the arms, not the pointer.
  // Space: start / stop-and-keep. X: redo (throw the take away). Only while
  // a session is open, and never while typing in a field.
  const runRef = useRef(run); runRef.current = run
  const apiRef = useRef(api); apiRef.current = api
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const cur = sRef.current
      const a = apiRef.current
      if (!a || !cur?.dataset) return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      if (e.code === 'Space') {
        e.preventDefault()
        void runRef.current(() => (cur.phase === 'recording' ? a.stopEpisode() : a.startEpisode()))
      } else if (e.key === 'x' || e.key === 'X') {
        if (cur.phase !== 'recording') return
        e.preventDefault()
        void runRef.current(() => a.redoEpisode())
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))
  const open = !!s?.dataset
  const recording = s?.phase === 'recording'
  // While recording, the last episode entry is the take in flight - its frame
  // count ticking is the only proof data is actually being captured. It is
  // not "kept" until stop saves it.
  // Never dereference the payload's shape: an older/other backend answering
  // /api/record/session with a session that has no `episodes` used to throw
  // during render, and a render that throws unmounts the whole app (JOURNEYS
  // #1). Missing simply means "no episodes yet".
  const episodes = Array.isArray(s?.episodes) ? s!.episodes : []
  const finished = recording ? episodes.slice(0, -1) : episodes
  const kept = finished.filter(e => !e?.discarded).length
  const liveFrames = recording && episodes.length > 0
    ? episodes[episodes.length - 1]?.frames ?? null
    : null

  // R1: the button's words depend on which recorder answered the probe.
  const openCopy = openActionCopy(api ? api.mock : null)

  return (
    <div className="train-sheet" role="dialog" aria-label="Record episodes">
      <div className="train-head">
        <h2>⏺ Record</h2>
        <button className="dock-min" onClick={onClose} aria-label="close">✕</button>
      </div>

      {api?.mock && (
        <div className="toast warn">
          This backend has no /api/record (older server?) — this is a rehearsal. Nothing is written to disk.
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
            <label className="field"><span>leader — you move this one</span>
              <select value={form.leader}
                      onChange={e => { setTouched(t => ({ ...t, leader: true })); set('leader', e.target.value) }}>
                <option value="">select…</option>
                {candidates.map(c => (
                  <option key={c.peer_id} value={c.peer_id}>{roleLabel(c)}</option>
                ))}
              </select>
            </label>
            <label className="field"><span>follower — gets recorded</span>
              <select value={form.follower}
                      onChange={e => { setTouched(t => ({ ...t, follower: true })); set('follower', e.target.value) }}>
                <option value="">select…</option>
                {candidates.map(c => (
                  <option key={c.peer_id} value={c.peer_id}>{roleLabel(c)}</option>
                ))}
              </select>
            </label>
            <label className="field"><span>episodes</span>
              <input inputMode="numeric" value={form.target_episodes}
                     onChange={e => set('target_episodes', e.target.value)} />
            </label>
          </div>
          {suggestion.basis === 'measured' && !suggestion.note && (
            <div className="train-msg rec-hint">
              paired from the servo buses — measured, not guessed from the names.
            </div>
          )}
          {suggestion.note && <div className="train-msg rec-hint">{suggestion.note}</div>}
          {problems.map(({ slot, msg }) => (
            <div key={slot} className="train-msg warn">⚠ {msg}</div>
          ))}
          {problems.length > 0 && (
            <label className="ackrow">
              <input type="checkbox" checked={ack} onChange={e => setAck(e.target.checked)} />
              <span>
                my arms really are wired that way — record anyway
                {problems.some(x => x.slot === 'leader')
                  ? ' (hand-moving a torqued arm can strip a gear: cut its power first)'
                  : ''}
              </span>
            </label>
          )}
          <div className="train-msg rec-hint">
            not sure which is which? the leader is the lighter 7.4V arm (no gearbox load —
            easy to move by hand); the follower is the stronger 12V arm that mirrors it.
          </div>
          {/* R1: the consequence is stated BEFORE the click, because this is the
              moment two real arms change state — not after, in a toast. */}
          <div className={`train-msg rec-hint${openCopy.cls ? ' warn' : ''}`}>{openCopy.hint}</div>
          <div className="train-actions">
            <button className={`btn go wide${openCopy.cls ? ` ${openCopy.cls}` : ''}`} type="submit"
                    aria-label={openCopy.aria}
                    disabled={busy || !api || !form.dataset.trim() || !form.task.trim()
                              || !form.leader || !form.follower || form.leader === form.follower
                              || (problems.length > 0 && !ack)}>
              {openCopy.label}
            </button>
          </div>
          {/* Only a real collision. Both slots start EMPTY now, and '' === '' was
              firing this warning at rest - scolding the operator before they had
              chosen anything. */}
          {!!form.leader && form.leader === form.follower &&
            <div className="train-msg">⚠ leader and follower must be different arms</div>}
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
                  ⏹ stop &amp; keep{liveFrames !== null ? ` · ${liveFrames}f` : ''}
                </button>
              </>
            )}
          </div>
          <div className="train-msg">
            {recording
              ? `recording — drive ${s.follower} with ${s.leader}, then stop (or redo to throw this one away)`
              : 'start when both arms are in position'}
          </div>
          <div className="train-msg rec-keys" aria-hidden="true">
            <kbd>space</kbd> {recording ? 'stop & keep' : 'start'} · <kbd>X</kbd> redo
          </div>
        </div>
      )}

      {open && s && <FollowerLive peer={peers.find(p => p.peer_id === s.follower)} recording={recording} />}

      {open && s && episodes.length > 0 && (
        <div className="rec-strip" role="list" aria-label="recorded episodes">
          {/* the last entry while recording is the take in flight - the live
              counter on the stop button represents it, not a card */}
          {finished.slice().reverse().map(ep => (
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
              ✓ finish dataset ({kept} kept{episodes.length - kept ? `, ${episodes.length - kept} discarded` : ''})
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
