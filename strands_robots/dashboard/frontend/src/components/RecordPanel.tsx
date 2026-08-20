import { useEffect, useRef, useState } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import type { Peer } from '../types'
import { getRecordApi, type RecordApi, type RecordSession } from '../lib/recordApi'
import { openActionCopy } from '../lib/recordAction'
import { sessionFreshness, staleSuffix } from '../lib/sessionFreshness'
import { api as httpGet, HttpError } from '../lib/endpoints'
import { recordFailure, type RecordActionKind } from '../lib/recordOutcome'
import { pairArms, roleLabel, contradiction, type RoleCandidate } from '../lib/armPairing'
import { noArmsVerdict, type RememberedBoard } from '../lib/noArms'
import { episodeTarget } from '../lib/episodeTarget'
import { fpsField, fpsSuggestion } from '../lib/recordFps'
import { nameVerdict, type KnownDataset } from '../lib/datasetName'
import { stoppedCameras, cameraWarning } from '../lib/cameraFreshness'
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
export default function RecordPanel(
  { peers, onClose, onDevices }: { peers: Peer[]; onClose: () => void; onDevices?: () => void },
) {
  const peerIds = peers.map(p => p.peer_id)
  const [api, setApi] = useState<RecordApi | null>(null)
  /* Q58: focus must land inside an overlay and go back to whatever opened it. */
  const sheetRef = useRef<HTMLDivElement | null>(null)
  useDialogFocus(sheetRef)
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
  // Q44: what the devices screen remembers, for the case where this screen has nothing to record
  // WITH. undefined = not asked yet, null = the request failed (which must not become the claim
  // "nothing is configured").
  const [boards, setBoards] = useState<RememberedBoard[] | null | undefined>(undefined)

  const suggestion = pairArms(candidates)
  // Replaces pairArms's "no arms on the mesh" note when there is nothing to record with: same fact,
  // with the way out. undefined (not asked yet) is passed through as a failed lookup rather than as
  // "nothing configured" - the honest reading while the request is in flight.
  const noArms = noArmsVerdict(peerIds.length, boards === undefined ? null : boards)
  const [form, setForm] = useState({
    dataset: '', task: '', leader: '', follower: '', target_episodes: '20', fps: '',
  })
  const [touched, setTouched] = useState({ leader: false, follower: false })
  // What the operator asked for, and whether we understood it — never a silent correction.
  const wanted = episodeTarget(form.target_episodes)
  // Q54: /api/record/open has always taken `fps` and this form never sent it, so every dataset
  // was stamped 30 while an SO-101 captures nearer 4 — and LeRobot derives timestamps from the
  // declaration, so the artifact claims the motion happened 7x faster than it did.
  const rate = fpsField(form.fps)
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

  // Q39: the backend refuses a taken dataset name before it parks the arms - but by then the
  // operator has picked a pair, aimed two cameras and pressed the button. The training picker's own
  // listing answers the question, so ask it once while the form is open and warn while there is
  // nothing at stake. Failure is silence: no evidence is not evidence of a problem.
  const [known, setKnown] = useState<KnownDataset[] | null>(null)
  useEffect(() => {
    let alive = true
    httpGet<{ datasets?: KnownDataset[] }>('/api/training/datasets?hub=false')
      .then(r => { if (alive) setKnown(r.datasets ?? []) })
      .catch(() => { if (alive) setKnown(null) })
    return () => { alive = false }
  }, [])
  const nameWarn = nameVerdict(form.dataset, known)
  const [closed, setClosed] = useState<string | null>(null)
  // When did a session read last ARRIVE (not: last get attempted)? A hung
  // request would otherwise keep this screen looking live forever.
  const [lastOkAt, setLastOkAt] = useState<number | null>(null)
  const [pollErr, setPollErr] = useState<string | null>(null)
  // Ticks once a second so the age on screen keeps growing even while a request
  // is stuck and no other state changes - the freeze is exactly the case where
  // the display must not sit still and look current.
  const [nowMs, setNowMs] = useState(() => Date.now())

  // The follower's cameras are what gets WRITTEN INTO THE DATASET, so a camera
  // that stopped publishing is a defect in the recording, not a cosmetic issue.
  // Measured on this fleet: arm-1's wrist last captured a frame 10.4 hours before
  // the record screen would have happily used it. The backend refuses this too
  // (409 + ignore_dead_cameras); asking here means the operator finds out before
  // they set up the scene, not after. Its own tick, separate from the wiring
  // acknowledgement: two different admissions must not share one checkbox.
  // Date.now() rather than the `nowMs` ticker: that one only runs while a session
  // is OPEN, and this question is asked on the form before it. Every peer event
  // re-renders this component, so the age stays current on its own.
  const followerPeer = peers.find(p => p.peer_id === form.follower)
  const deadCams = stoppedCameras(followerPeer?.cameras, Date.now() / 1000)
  const camWarning = cameraWarning(deadCams, { peerId: form.follower })
  const [camAck, setCamAck] = useState(false)

  // The measured roles. Managed children carry them keyed by peer id, so this is
  // one request, not one per arm. A failure here is not worth a banner: the
  // pickers simply stay unopinionated (basis 'none' says so out loud).
  useEffect(() => {
    let alive = true
    httpGet<{
      managed?: Record<string, RoleCandidate & { peer_id: string; alive?: boolean; port?: string }>
      serial_ports?: { device: string; remembered?: { peer_id: string } | null }[]
    }>('/api/devices')
      .then(doc => {
        if (!alive) return
        const next: Record<string, RoleCandidate> = {}
        for (const m of Object.values(doc.managed ?? {})) {
          if (m?.peer_id) next[m.peer_id] = { peer_id: m.peer_id, role: m.role, role_volts: m.role_volts }
        }
        setRoles(next)
        // Same document already in hand - no second request for the empty-state route.
        const claimed = new Set(Object.values(doc.managed ?? {})
          .filter((m: any) => m?.alive && m?.port).map((m: any) => m.port as string))
        setBoards((doc.serial_ports ?? [])
          .filter(p => p.remembered?.peer_id)
          .map(p => ({ peer_id: p.remembered!.peer_id, claimed: claimed.has(p.device) })))
      })
      .catch(() => { setBoards(null); /* unmeasured is a valid state, not an error */ })
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
      a.session()
        .then(sess => { if (alive) { setS(sess); setLastOkAt(Date.now()) } })
        .catch(e => alive && setErr(String(e)))
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
      setNowMs(Date.now())
      if (pending) return
      pending = true
      api.session()
        .then(sess => { if (alive) { setS(sess); setLastOkAt(Date.now()); setPollErr(null) } })
        // One lost tick IS a blip. Several in a row means the numbers on screen
        // are a photograph, and the frame count - the only proof an episode is
        // being captured - has quietly stopped meaning anything.
        .catch(e => { if (alive) setPollErr(e instanceof Error ? e.message : String(e)) })
        .finally(() => { pending = false })
    }, 1000)
    return () => { alive = false; clearInterval(t) }
  }, [api, s?.dataset])

  const run = async (fn: () => Promise<RecordSession>, kind: RecordActionKind) => {
    if (busy) return
    setBusy(true); setErr(null)
    // An action's own answer is a fresh read of the session: it counts.
    try { setS(await fn()); setLastOkAt(Date.now()); setPollErr(null) }
    catch (e) {
      // A thrown record action is NOT proof it did not happen: the request may
      // have reached the recorder, acted, and lost its answer (lib/recordOutcome).
      const v = recordFailure({
        kind,
        status: e instanceof HttpError ? e.status : 0,
        message: e instanceof Error ? e.message : String(e),
      })
      setErr(v.text)
      // Hand off to the observer that knows. The 1s poll only runs once a session
      // exists, so an `open` that MAY have landed would otherwise leave nothing
      // watching at all - the one case that most needs a read.
      if (v.ambiguous && api) {
        try {
          setS(await api.session()); setLastOkAt(Date.now()); setPollErr(null)
        } catch { /* the message already says the read is what to watch */ }
      }
    }
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
        // The key does two different things, so it must report two different
        // consequences: a lost stop may have saved the take, a lost start may
        // already be recording.
        const recording = cur.phase === 'recording'
        void runRef.current(() => (recording ? a.stopEpisode() : a.startEpisode()), recording ? 'stop' : 'start')
      } else if (e.key === 'x' || e.key === 'X') {
        if (cur.phase !== 'recording') return
        e.preventDefault()
        void runRef.current(() => a.redoEpisode(), 'redo')
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
  // Are those numbers still live? The poller used to swallow its failures, so a
  // frozen frame count read exactly like a ticking one.
  const fresh = sessionFreshness({ lastOkAtMs: lastOkAt, nowMs, lastError: pollErr, recording })

  // R1: the button's words depend on which recorder answered the probe.
  const openCopy = openActionCopy(api ? api.mock : null)

  return (
    <div ref={sheetRef} className="train-sheet" role="dialog" aria-label="Record episodes">
      <div className="train-head">
        <h2>⏺ Record</h2>
        <button className="dock-min" onClick={onClose} aria-label="close">✕</button>
      </div>

      {api?.mock && (
        <div className="toast warn">
          This backend has no /api/record (older server?) — this is a rehearsal. Nothing is written to disk.
        </div>
      )}

      {/* Q40: a session the dashboard died inside used to be silence — an empty form over a
          half-written dataset, with both arms left despawned by the parking step. The server sends
          the sentence (it has the facts); this renders it, and renders the next actions as WORDS.
          No button here deletes a dataset: destroying an hour of hand-guiding is not a thing a
          screen should offer next to a form the operator is about to fill in. */}
      {!open && s?.interrupted && (
        <div className="artifact-hold" role="status">
          <div>⏹ {s.interrupted.text}</div>
          <ul className="hold-next">
            {s.interrupted.next.map(n => <li key={n}>{n}</li>)}
          </ul>
        </div>
      )}
      {!open && s && (
        <form className="train-form" onSubmit={e => {
          e.preventDefault()
          void run(() => api!.open({
            dataset: form.dataset.trim(), task: form.task.trim(),
            leader: form.leader, follower: form.follower,
            target_episodes: wanted.value,
            fps: rate.value,
            // Only ever sent when the operator ticked the box in front of the
            // named camera and its age - never a default, never remembered.
            ...(camWarning && camAck ? { ignore_dead_cameras: true } : {}),
          }), 'open')
        }}>
          <label className="field"><span>dataset (name or hf repo id)</span>
            <input value={form.dataset} placeholder="cagatay/so101-pick-cube"
                   onChange={e => set('dataset', e.target.value)} />
          </label>
          {/* Not a validator: it never blocks the submit and never rewrites the field. The
              suggestion is one tap because someone naming a dataset twice usually wants "-2",
              not a lecture. */}
          {nameWarn && (
            <div className="train-msg" role="status">
              ⚠ {nameWarn.message}
              {nameWarn.suggestion && (
                <> <button type="button" className="btn ghost"
                           onClick={() => set('dataset', nameWarn.suggestion!)}>
                  use {nameWarn.suggestion}
                </button></>
              )}
            </div>
          )}
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
            <label className="field"><span>fps</span>
              <input inputMode="numeric" value={form.fps} placeholder="30"
                     aria-invalid={!!rate.problem} aria-describedby="rec-fps-say"
                     onChange={e => set('fps', e.target.value)} />
              {/* The rate the dataset DECLARES. Empty = the backend's 30, said out loud rather
                  than left as a hidden default nobody could see or change. */}
              <span id="rec-fps-say" className={`fieldsay${rate.problem ? ' bad' : ''}`}>
                {rate.problem ?? rate.note ?? 'timestamps are derived from this — match your real capture rate'}
              </span>
            </label>
            <label className="field"><span>episodes</span>
              <input inputMode="numeric" value={form.target_episodes}
                     aria-invalid={!!wanted.problem} aria-describedby="rec-episodes-say"
                     onChange={e => set('target_episodes', e.target.value)} />
              {/* The old expression was Math.max(1, Number(raw) || 20): a typo like "3o" opened a
                  TWENTY-episode session, "0" and "-5" became 1, and nothing said so — while the
                  arms had already left the fleet and the follower was energised. */}
              <span id="rec-episodes-say" className={`fieldsay${wanted.problem ? ' bad' : ''}`}>
                {wanted.problem ?? wanted.note ?? ''}
              </span>
            </label>
          </div>
          {suggestion.basis === 'measured' && !suggestion.note && (
            <div className="train-msg rec-hint">
              paired from the servo buses — measured, not guessed from the names.
            </div>
          )}
          {/* Q44: "no arms on the mesh" was true and a dead end. After a restart the arms are not
              unplugged, just not running — and the devices screen knows them by USB serial with a
              one-click respawn. Name that route here instead of making the operator hunt for a
              feature we shipped. A sentence plus an offer, never a redirect. */}
          {noArms
            ? <div className="train-msg rec-hint" role="status">
                {noArms.text}
                {noArms.offerDevices && onDevices && (
                  <> <button type="button" className="btn ghost" onClick={onDevices}>
                    open the devices screen
                  </button></>
                )}
              </div>
            : suggestion.note && <div className="train-msg rec-hint">{suggestion.note}</div>}
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
          {camWarning && (
            <div className="train-msg warn">⚠ {camWarning}</div>
          )}
          {camWarning && (
            <label className="ackrow">
              <input type="checkbox" checked={camAck} onChange={e => setCamAck(e.target.checked)} />
              <span>
                record without {deadCams.length > 1 ? 'those cameras' : `the ${deadCams[0].camera} camera`} anyway
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
                              || !!wanted.problem || !!rate.problem
                              || (problems.length > 0 && !ack)
                              || (!!camWarning && !camAck)}>
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

      {/*
        A camera that never opened, said out loud BEFORE the episodes are
        collected. role=alert (not aria-live=polite) because it changes what
        the operator should do next: every episode will look successful and the
        finished dataset will have no image channel to train on.
      */}
      {open && s?.camera_notice && (
        <div className="train-msg warn rec-camera-notice" role="alert">
          ⚠ {s.camera_notice.message}
        </div>
      )}

      {/* The captured rate, when it disagrees with the rate this dataset
          DECLARES. LeRobot timestamps a frame as frame_index/fps, so the gap
          leaves no trace in the artifact — if it is not said here, while the
          operator can still stop and re-open, it is never said at all. A
          notice, never a block: they are holding a leader arm. */}
      {open && s?.fps_notice && (
        <div className="train-msg warn rec-fps-notice" role="alert">
          ⚠ {s.fps_notice.detail}
          {/* Q54: the notice used to end here — a warning about a number no screen could change.
              The remedy is future-tense on purpose: this session's episodes are already stamped,
              and offering to "fix" them would be one lie deeper. */}
          {(() => {
            const sug = fpsSuggestion(s.fps_notice)
            if (!sug) return null
            return (
              <div className="rec-fps-fix">
                <button className="btn ghost" type="button"
                        onClick={() => set('fps', sug.fps)}>{sug.label}</button>
                <span className="fieldsay">{sug.why}</span>
              </div>
            )
          })()}
        </div>
      )}

      {/*
        The follower has not moved for the whole window. role=alert like the two
        above, and for the same reason: every counter looks perfect while the
        dataset being written is one pose repeated, so if this is not said while
        the operator can still redo the episode it is discovered at training time
        or never. A NOTICE - holding still is legitimate (lining the arms up,
        holding a grasp), and stopping recording here would throw away real
        episodes to prevent a suspicion.
      */}
      {open && s?.motion_notice && (
        <div className="train-msg warn rec-motion-notice" role="alert">
          ⚠ {s.motion_notice.message}
        </div>
      )}

      {open && s && (
        <div className="train-form">
          <div className="rec-counter" aria-live="polite">
            <b>{kept}</b><span> / {s.target_episodes} episodes</span>
            <span className="rec-task">{s.dataset} — “{s.task}”</span>
            {/* declared → measured, side by side: the pair is the point */}
            <span className="rec-rate" title="declared fps vs the rate actually captured">
              {s.fps} fps
              {s.fps_achieved != null && (
                <span className={s.fps_notice ? 'rec-rate-bad' : 'rec-rate-ok'}>
                  {' '}· {s.fps_achieved} captured
                </span>
              )}
              {/* A healthy-looking rate is exactly what a frozen arm produces, so
                  the pair itself carries the doubt rather than only the banner. */}
              {s.motion_notice && (
                <span className="rec-rate-bad" title={s.motion_notice.message}>
                  {' '}· not moving
                </span>
              )}
            </span>
          </div>
          <div className="train-actions">
            {!recording ? (
              <button className="btn go wide" onClick={() => void run(() => api!.startEpisode(), 'start')} disabled={busy}>
                ⏺ start episode {kept + 1}
              </button>
            ) : (
              <>
                <button className="btn wide" onClick={() => void run(() => api!.redoEpisode(), 'redo')} disabled={busy}>
                  ↺ redo
                </button>
                <button className="btn go wide rec-live" onClick={() => void run(() => api!.stopEpisode(), 'stop')} disabled={busy}>
                  ⏹ stop &amp; keep{liveFrames !== null ? ` · ${liveFrames}f${staleSuffix(fresh)}` : ''}
                </button>
              </>
            )}
          </div>
          {fresh.text && <div className={`toast ${fresh.tone === 'bad' ? 'bad' : 'warn'}`}>{fresh.text}</div>}
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
                : <button className="btn ghost" onClick={() => void run(() => api!.discard(ep.index), 'discard')} disabled={busy}>
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
            /* Q56: this was a free-text "hub repo id" box, and a LeRobotDataset can only publish
               under the repo_id it was created with — there is no argument that redirects the push.
               So the box could only produce a refusal at the end of the session. It now states
               where the push goes, and the two things the operator cannot see: the repo is public
               unless their namespace defaults otherwise, and nothing in this dashboard can retry a
               failed push (the session is gone once it closes). */
            <p className="hint">
              publishes as <code>{s.dataset ?? '(unnamed)'}</code> — a dataset can only be pushed
              under the name it was recorded with. It will be <b>public</b> unless your Hub namespace
              defaults to private, and if the push fails the episodes stay on this machine: finishing
              closes the session, so a retry is a <code>huggingface-cli</code> job.
            </p>
          )}
          <div className="train-actions">
            <button className="btn wide" disabled={busy} onClick={() => {
              void (async () => {
                setBusy(true); setErr(null)
                try {
                  const r = await api!.close(upload ? { upload } : {})
                  setClosed(r.detail ?? (r.ok ? `dataset finished with ${kept} episode(s)` : 'close failed'))
                  setS(await api!.session())
                } catch (e) {
                  // Finishing reaches outside this machine when upload is ticked:
                  // a lost answer may mean the dataset is already on the Hub.
                  const v = recordFailure({
                    kind: 'close',
                    status: e instanceof HttpError ? e.status : 0,
                    message: e instanceof Error ? e.message : String(e),
                  })
                  setErr(v.text)
                  if (v.ambiguous) {
                    try { setS(await api!.session()); setLastOkAt(Date.now()); setPollErr(null) } catch { /* the message says what to watch */ }
                  }
                }
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
