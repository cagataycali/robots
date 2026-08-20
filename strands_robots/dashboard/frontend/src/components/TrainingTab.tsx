import { useEffect, useRef, useState } from 'react'
import { trainingFreshness } from '../lib/trainingFreshness'
import { api, post, HttpError } from '../lib/endpoints'
import { sideEffectVerdict, type SideEffectKind } from '../lib/submitOutcome'
import LossSpark from './LossSpark'
import { pushLoss, fmtStep, type LossPoint } from '../lib/lossTrace'
import { setDeployIntent } from '../lib/deployIntent'
import { datasetKey as dsKey, selectDataset, selectionKey, replayable, type DatasetRow } from '../lib/datasetSelection'
import { datasetHint, isCurrentResponse } from '../lib/datasetHint'

// One row is either LOCAL (has a `root` path, trains offline) or from the HUB
// (no root, trains from repo_id after a download). lib/datasetSelection owns
// that distinction and the rule that exactly one field reaches the trainer.
type Dataset = DatasetRow
interface Job { job_id: string; provider: string; dataset?: string; base_model?: string; output_dir?: string; steps?: number; submitted_at?: number }
interface JobStatus { status: string; data: { status?: string; metrics?: Record<string, unknown> }; text: string }

/**
 * Training tab - submit / monitor / export policy training jobs.
 *
 * Backed by train_policy, the one workflow tool with structured JSON
 * results (job_id, status, metrics) - no prose parsing. Dataset picker
 * scans local LeRobotDataset roots; the trained checkpoint feeds straight
 * back into the run form's checkpoint search (record → train → deploy).
 */
// Same family-name heuristic the backend's checkpoint search uses - only a
// PREFILL for the run form's policy_type field, never a decision.
function guessPolicyType(baseModel: string | undefined): string | null {
  const m = (baseModel ?? '').replace(/_/g, '-').match(/\b(smolvla|act|diffusion|pi0-fast|pi05|pi0|tdmpc|vqbet)\b/i)
  return m ? m[1].toLowerCase().replace('-', '_').replace('pi0fast', 'pi0_fast') : null
}

export default function TrainingTab({ onClose }: { onClose: () => void }) {
  const [trainers, setTrainers] = useState<string[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [statuses, setStatuses] = useState<Record<string, JobStatus>>({})
  // WHEN each status was last read, and how many reads have failed since - a
  // swallowed poll error used to leave a dead run rendering as healthy progress.
  const [polledAt, setPolledAt] = useState<Record<string, number>>({})
  const [pollFail, setPollFail] = useState<Record<string, { n: number; msg: string }>>({})
  const [nowS, setNowS] = useState(() => Date.now() / 1000)
  const [traces, setTraces] = useState<Record<string, LossPoint[]>>({})
  const [form, setForm] = useState({ provider: 'lerobot_local', dataset_root: '', dataset_repo_id: '', base_model: 'lerobot/smolvla_base', output_dir: '', steps: '10000', method: 'lora' })
  // R6: the picker searches the Hub as you type. `dsProblem` is the HUB half's
  // verdict only — "no matches" is a real answer and must not wear an outage's
  // clothes, so an empty list with problem===null says something different from
  // an empty list with a reason.
  const [dsQuery, setDsQuery] = useState('')
  const [dsProblem, setDsProblem] = useState<string | null>(null)
  // WHICH query the rows and the verdict on screen were measured for. Hub round
  // trips are not ordered, so "the last response" is not "the current answer" -
  // see lib/datasetHint.ts.
  const [dsShownQuery, setDsShownQuery] = useState<string | null>(null)
  const dsSeq = useRef(0)
  const [dsAuth, setDsAuth] = useState<{ authenticated?: boolean; user?: string | null; detail?: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const refresh = async () => {
    const seq = ++dsSeq.current
    try {
      const [t, d, j] = await Promise.all([
        api('/api/training/trainers'),
        api(`/api/training/datasets?q=${encodeURIComponent(dsQuery)}`),
        api('/api/training/jobs'),
      ])
      setTrainers(t.trainers ?? [])
      setJobs((j.jobs ?? []).slice().reverse())
      // A keystroke may have fired a newer dataset search while this was in
      // flight; the newer question owns the list.
      if (isCurrentResponse(seq, dsSeq.current)) {
        setDatasets(d.datasets ?? [])
        setDsProblem(d.problem ?? null)
        setDsAuth(d.hf_auth ?? null)
        setDsShownQuery(dsQuery)
      }
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
  }
  useEffect(() => { refresh() }, [])

  // Type-ahead: datasets only (re-polling jobs on every keystroke would be
  // rude to a running job's status endpoint). 250ms because each miss is a Hub
  // round trip; the backend caches a hit for 5 minutes, never a failure.
  useEffect(() => {
    const t = setTimeout(async () => {
      const seq = ++dsSeq.current
      const asked = dsQuery
      try {
        const d = await api(`/api/training/datasets?q=${encodeURIComponent(asked)}`)
        // Out-of-order Hub answers: a slow reply for "so" must not overwrite the
        // rows for "so101" that the user is about to pick from.
        if (!isCurrentResponse(seq, dsSeq.current)) return
        setDatasets(d.datasets ?? [])
        setDsProblem(d.problem ?? null)
        setDsAuth(d.hf_auth ?? null)
        setDsShownQuery(asked)
      } catch (e) {
        if (!isCurrentResponse(seq, dsSeq.current)) return
        setDsProblem(`search failed: ${(e as any)?.message ?? e}`)
        setDsShownQuery(asked)
      }
    }, 250)
    return () => clearTimeout(t)
  }, [dsQuery])

  // poll running job statuses every 5s
  useEffect(() => {
    const id = setInterval(async () => {
      for (const job of jobs.slice(0, 5)) {
        if (!job.job_id) continue
        try {
          const s = await api(`/api/training/status?provider=${job.provider}&job_id=${encodeURIComponent(job.job_id)}`)
          setStatuses(prev => ({ ...prev, [job.job_id]: s }))
          setPolledAt(prev => ({ ...prev, [job.job_id]: Date.now() / 1000 }))
          setPollFail(prev => (prev[job.job_id] ? { ...prev, [job.job_id]: { n: 0, msg: '' } } : prev))
          const m = s?.data?.metrics as Record<string, unknown> | undefined
          if (m) setTraces(prev => ({
            ...prev,
            [job.job_id]: pushLoss(prev[job.job_id] ?? [], m.latest_step, m.latest_loss),
          }))
        } catch (e) {
          // ONE failed poll is transient; an unbounded swallow is how a dead run
          // keeps rendering "running" at 4.7k/10k steps forever. Count them, and
          // let the card say the numbers went old.
          const msg = String((e as any)?.message ?? e).slice(0, 120)
          setPollFail(prev => ({ ...prev, [job.job_id]: { n: (prev[job.job_id]?.n ?? 0) + 1, msg } }))
        }
      }
    }, 5000)
    return () => clearInterval(id)
  }, [jobs])

  // The age must advance while NOTHING arrives, so it cannot be driven by the
  // poll that stopped.
  useEffect(() => {
    const id = setInterval(() => setNowS(Date.now() / 1000), 5000)
    return () => clearInterval(id)
  }, [])

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))

  /**
   * A failed request that STARTS something is not the same as one that did not
   * happen: a rejected fetch covers "never left this machine" and "ran, then
   * lost the answer". Saying `⚠ <message>` invited a second press — a second
   * multi-hour run, a second recorder on one dataset, a second peer driving the
   * same arm. `refresh()` runs on the ambiguous branch so the list that KNOWS
   * gets a chance to answer.
   */
  const failed = (kind: SideEffectKind, e: unknown) => {
    const v = sideEffectVerdict({
      kind,
      status: e instanceof HttpError ? e.status : 0,
      message: (e as any)?.message ?? String(e),
    })
    setMsg(v.text)
    if (v.delivered === 'unknown') refresh()
  }

  const submit = async (validateOnly: boolean) => {
    setBusy(true); setMsg(null)
    const body = {
      provider: form.provider,
      dataset_root: form.dataset_root || undefined,
      dataset_repo_id: form.dataset_repo_id || undefined,
      base_model: form.base_model || undefined,
      output_dir: form.output_dir || undefined,
      steps: Number(form.steps) || 10000,
      method: form.method || undefined,
    }
    try {
      const j = await post(validateOnly ? '/api/training/validate' : '/api/training/submit', body)
      setMsg(j.status === 'success' ? `✓ ${j.text?.slice(0, 200)}` : `✗ ${j.text?.slice(0, 300)}`)
      if (!validateOnly && j.status === 'success') refresh()
    } catch (e) {
      // A validate is read-only: it cannot leave a run behind, so it must not
      // claim it might have.
      failed(validateOnly ? 'export' : 'training', e)
    }
    setBusy(false)
  }

  const [collect, setCollect] = useState({ dataset_root: '', instruction: 'pick up the red cube', n_episodes: '5', duration: '10', robot_name: 'so101' })
  const [showCollect, setShowCollect] = useState(false)

  const submitCollect = async () => {
    if (!collect.dataset_root.trim()) return
    setBusy(true); setMsg(null)
    try {
      const j = await post('/api/collect', {
        dataset_root: collect.dataset_root,
        instruction: collect.instruction,
        n_episodes: Number(collect.n_episodes) || 5,
        duration: Number(collect.duration) || 10,
        robot_name: collect.robot_name,
      })
      setMsg(j.peer_id
        ? `▶ collecting ${j.n_episodes} episodes as ${j.peer_id} — watch it in the fleet grid; dataset appears below when done`
        : `⚠ ${JSON.stringify(j).slice(0, 200)}`)
      if (j.peer_id) setTimeout(refresh, 15000)
    } catch (e) { failed('collect', e) }
    setBusy(false)
  }

  const replay = async (d: Dataset) => {
    // Replay reads episode 0 off the disk. A Hub row is not on this disk yet,
    // so replaying it would fail somewhere deep in a loader; say so here.
    if (!d.root) {
      setMsg(`⚠ ${d.repo_id} is on the Hub, not on this machine — train with it (the trainer downloads it) or clone it locally to replay`)
      return
    }
    setBusy(true); setMsg(null)
    try {
      const j = await post('/api/replay', { repo_id: d.repo_id, root: d.root, episode: 0 })
      setMsg(j.peer_id
        ? `▶ replaying ${d.repo_id} ep0 as ${j.peer_id} — watch it in the fleet grid`
        : `⚠ ${JSON.stringify(j).slice(0, 200)}`)
    } catch (e) { failed('replay', e) }
    setBusy(false)
  }

  const exportCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      setMsg(j.status === 'success' ? `✓ ${j.text?.slice(0, 250)}` : `✗ ${j.text?.slice(0, 250)}`)
    } catch (e) { failed('export', e) }
    setBusy(false)
  }

  // "Deploy" cannot start a policy from here - the run form is per-robot and
  // a policy moves a real arm. So deploy = export (to get the honest loadable
  // path from the trainer, never a guessed directory), stamp a deploy intent,
  // and send the user to a robot card whose run form will prefill from it and
  // WAIT for them to press Run.
  const deployCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      const ckpt = j?.data?.exported_model
      if (j.status !== 'success' || typeof ckpt !== 'string' || !ckpt) {
        setMsg(`✗ nothing deployable: ${j.text?.slice(0, 200) ?? 'export returned no artifact path'}`)
      } else {
        setDeployIntent({
          checkpoint: ckpt,
          policy_type: guessPolicyType(job.base_model),
          source: `training job ${job.job_id} (${job.base_model || job.provider})`,
        })
        setMsg('🚀 checkpoint staged — close this sheet and open a robot\u2019s run form: it will be prefilled, and nothing runs until you press Run there')
      }
    } catch (e) { failed('export', e) }
    setBusy(false)
  }


  // The form re-told as one sentence. A grid of fields answers "what can I
  // set"; the sentence answers the question that actually matters before a
  // multi-hour job: "what did I just ask for?"
  const datasetPicked = form.dataset_root || form.dataset_repo_id
  const datasetLabel = selectDataset(datasets, selectionKey(form)).label || null
  const story = datasetPicked
    ? `Fine-tune ${form.base_model || 'lerobot/smolvla_base'} on ${datasetLabel ?? datasetPicked} for ${fmtStep(Number(form.steps) || 10000)} steps (${form.method}), saving to ${form.output_dir || '…pick an output dir'}.`
    : 'Pick a dataset to begin — the plan reads back here before anything runs.'

  return (
    <div className="train-sheet">
      <div className="train-head">
        <h2>🎓 Training</h2>
        <button className="dock-min" onClick={onClose}>✕</button>
      </div>

      <div className="train-form">
        <p className={`train-story${datasetPicked ? '' : ' empty'}`}>{story}</p>
        <label className="field"><span>provider</span>
          <select value={form.provider} onChange={e => set('provider', e.target.value)} disabled={busy}>
            {trainers.map(t => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label className="field"><span>dataset</span>
          <input value={dsQuery} onChange={e => setDsQuery(e.target.value)} disabled={busy}
                 placeholder="search this machine and the Hub — e.g. pusht, so101, your org" />
          {/* Selecting sets EXACTLY ONE of dataset_root / dataset_repo_id: a
              local dataset trains from its path, a Hub one from its repo id,
              and sending both would leave the trainer to pick for you. */}
          <select value={selectionKey(form)}
                  onChange={e => {
                    const sel = selectDataset(datasets, e.target.value)
                    setForm(f => ({ ...f, dataset_root: sel.dataset_root, dataset_repo_id: sel.dataset_repo_id }))
                  }} disabled={busy}>
            <option value="">— pick a dataset —</option>
            {datasets.filter(d => d.local !== false).length > 0 && (
              <optgroup label="on this machine">
                {datasets.filter(d => d.local !== false).map(d => (
                  <option key={dsKey(d)} value={dsKey(d)}>
                    {d.repo_id} ({d.total_episodes ?? '?'} eps{d.robot_type && d.robot_type !== 'unknown' ? `, ${d.robot_type}` : ''})
                  </option>
                ))}
              </optgroup>
            )}
            {datasets.filter(d => d.local === false).length > 0 && (
              <optgroup label="HuggingFace Hub — downloaded when training starts">
                {datasets.filter(d => d.local === false).map(d => (
                  <option key={dsKey(d)} value={dsKey(d)}>
                    {d.repo_id}{d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ''}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
          {/* ONE verdict, so the rows, the failure and the sentence cannot
              describe three different moments (lib/datasetHint.ts). */}
          {(() => {
            const h = datasetHint({
              query: dsQuery, shownQuery: dsShownQuery, count: datasets.length,
              problem: dsProblem, anonymous: dsAuth?.authenticated === false,
              authDetail: dsAuth?.detail ?? null,
            })
            return (
              <>
                {h.text && <span className={`hint${h.tone === 'warn' ? ' warn' : ''}`} role="status" aria-live="polite">{h.text}</span>}
                {h.auth && <span className="hint">{h.auth}</span>}
              </>
            )
          })()}
        </label>
        <label className="field"><span>base model</span>
          <input value={form.base_model} onChange={e => set('base_model', e.target.value)} disabled={busy} placeholder="lerobot/smolvla_base" />
        </label>
        <label className="field"><span>output dir</span>
          <input value={form.output_dir} onChange={e => set('output_dir', e.target.value)} disabled={busy} placeholder="/tmp/my_policy_ckpt" />
        </label>
        <div className="train-row">
          <label className="field"><span>steps</span>
            <input type="number" value={form.steps} onChange={e => set('steps', e.target.value)} disabled={busy} />
          </label>
          <label className="field"><span>method</span>
            <select value={form.method} onChange={e => set('method', e.target.value)} disabled={busy}>
              <option value="lora">lora</option>
              <option value="full">full</option>
            </select>
          </label>
        </div>
        <div className="train-actions">
          <button className="btn ghost" onClick={() => submit(true)} disabled={busy}>✓ validate</button>
          <button className="btn go wide" onClick={() => submit(false)} disabled={busy || !datasetPicked || !form.output_dir}>▶ train</button>
        </div>
        {msg && <div className="train-msg">{msg}</div>}
      </div>

      <div className="train-form">
        <button className="btn ghost" onClick={() => setShowCollect(s => !s)}>
          {showCollect ? '▾' : '▸'} 📹 collect new dataset (sim rollouts)
        </button>
        {showCollect && (
          <>
            <label className="field"><span>dataset root (path)</span>
              <input value={collect.dataset_root} placeholder="/tmp/my_demos"
                onChange={e => setCollect(c => ({ ...c, dataset_root: e.target.value }))} disabled={busy} />
            </label>
            <label className="field"><span>instruction</span>
              <input value={collect.instruction}
                onChange={e => setCollect(c => ({ ...c, instruction: e.target.value }))} disabled={busy} />
            </label>
            <div className="train-row">
              <label className="field"><span>episodes</span>
                <input type="number" value={collect.n_episodes}
                  onChange={e => setCollect(c => ({ ...c, n_episodes: e.target.value }))} disabled={busy} />
              </label>
              <label className="field"><span>sec/episode</span>
                <input type="number" value={collect.duration}
                  onChange={e => setCollect(c => ({ ...c, duration: e.target.value }))} disabled={busy} />
              </label>
              <label className="field"><span>robot</span>
                <input value={collect.robot_name}
                  onChange={e => setCollect(c => ({ ...c, robot_name: e.target.value }))} disabled={busy} />
              </label>
            </div>
            <div className="train-actions">
              <button className="btn go wide" onClick={submitCollect} disabled={busy || !collect.dataset_root.trim()}>
                📹 collect
              </button>
            </div>
          </>
        )}
      </div>

      <div className="train-jobs">
        <h3>Datasets</h3>
        {datasets.length === 0 && (
          <div className="dock-hint">
            {dsShownQuery !== null && dsShownQuery.trim() !== dsQuery.trim()
              ? `Searching for “${dsQuery.trim()}”…`
              : dsProblem
                ? `No local LeRobotDatasets, and the Hub could not be searched — ${dsProblem}`
                : dsQuery
                  ? `Nothing matches “${dsQuery}” here or on the Hub.`
                  : 'No LeRobotDatasets on this machine — type above to search the Hub, or record one in Collect.'}
          </div>
        )}
        {datasets.map(d => (
          <div className="train-job" key={dsKey(d)}>
            <div className="train-job-head">
              <b>{d.repo_id}</b>
              <span className="jstate">
                {d.root
                  ? `${d.total_episodes ?? '?'} eps · ${d.fps ?? '?'} fps`
                  : `Hub${d.downloads ? ` · ${d.downloads.toLocaleString()} downloads` : ''}`}
              </span>
            </div>
            <div className="train-job-actions">
              {/* Replay reads episode 0 off this disk, so it is offered only for
                  what is actually here — a disabled button with the reason beats
                  a click that dies inside a dataset loader. */}
              <button className="btn ghost" onClick={() => replay(d)} disabled={busy || !replayable(d).ok}
                title={replayable(d).reason}>
                🎬 replay in sim
              </button>
            </div>
          </div>
        ))}

        <h3>Jobs</h3>
        {jobs.length === 0 && <div className="dock-hint">No training jobs yet.</div>}
        {jobs.map(job => {
          const st = statuses[job.job_id]
          const state = st?.data?.status ?? '…'
          const fresh = trainingFreshness({
            polledAtS: polledAt[job.job_id], nowS,
            failures: pollFail[job.job_id]?.n, error: pollFail[job.job_id]?.msg || null,
            state: st?.data?.status ?? null,
          })
          return (
            <div className={`train-job${fresh.stale ? ' stalefeed' : ''}`} key={job.job_id ?? Math.random()}>
              <div className="train-job-head">
                <b>{job.provider}</b>
                {/* The chip is the word an operator trusts for hours, so it says
                    how old that word is - always, not only when it goes stale. */}
                <span className={`jstate ${state}`} title={fresh.title}>{state}</span>
                {fresh.stale && <span className="jstale" title={fresh.title}>
                  as of {fresh.ageS != null && fresh.ageS < 90 ? `${Math.round(fresh.ageS)}s` : `${Math.round((fresh.ageS ?? 0) / 60)}m`} ago
                </span>}
              </div>
              {fresh.note && <div className="train-msg warn">{fresh.note}</div>}
              <div className="train-job-meta">
                {job.dataset?.split('/').slice(-2).join('/')} → {job.output_dir} · {job.steps} steps
              </div>
              {(() => {
                const m = st?.data?.metrics as Record<string, any> | undefined
                if (!m || Object.keys(m).length === 0) return null
                const step = typeof m.latest_step === 'number' ? m.latest_step : null
                const total = typeof job.steps === 'number' ? job.steps : Number(job.steps) || null
                return (
                  <>
                    {step !== null && total ? (
                      <div className="train-progress" role="progressbar"
                           aria-valuemin={0} aria-valuemax={total} aria-valuenow={Math.min(step, total)}>
                        <div className="train-progress-fill"
                             style={{ width: `${Math.min(100, (step / total) * 100)}%` }} />
                        <span className="train-progress-label">{fmtStep(step)} / {fmtStep(total)} steps</span>
                      </div>
                    ) : null}
                    <LossSpark trace={traces[job.job_id] ?? []} />
                    {m.latest_loss !== undefined && !Number.isFinite(m.latest_loss) && (
                      <div className="train-msg">⚠ loss is NaN — the run is executing but NOT learning (check LR / data)</div>
                    )}
                    {m.liveness_ok === false && state === 'running' && (
                      <div className="train-msg">⚠ no step lines in the log yet — still warming up, or stalled</div>
                    )}
                  </>
                )
              })()}
              <div className="train-job-actions">
                <button className="btn ghost" onClick={() => exportCkpt(job)} disabled={busy}>📦 export checkpoint</button>
                {state === 'success' && (
                  <button className="btn ghost" onClick={() => deployCkpt(job)} disabled={busy}
                          title="stages this checkpoint into a robot's run form — never starts it">🚀 deploy…</button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
