import { useEffect, useState } from 'react'
import { api, post } from '../lib/endpoints'
import LossSpark from './LossSpark'
import { pushLoss, fmtStep, type LossPoint } from '../lib/lossTrace'
import { setDeployIntent } from '../lib/deployIntent'

interface Dataset { root: string; repo_id: string; total_episodes?: number; robot_type?: string; fps?: number }
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
  const [traces, setTraces] = useState<Record<string, LossPoint[]>>({})
  const [form, setForm] = useState({ provider: 'lerobot_local', dataset_root: '', base_model: 'lerobot/smolvla_base', output_dir: '', steps: '10000', method: 'lora' })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const refresh = async () => {
    try {
      const [t, d, j] = await Promise.all([
        api('/api/training/trainers'),
        api('/api/training/datasets'),
        api('/api/training/jobs'),
      ])
      setTrainers(t.trainers ?? [])
      setDatasets(d.datasets ?? [])
      setJobs((j.jobs ?? []).slice().reverse())
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
  }
  useEffect(() => { refresh() }, [])

  // poll running job statuses every 5s
  useEffect(() => {
    const id = setInterval(async () => {
      for (const job of jobs.slice(0, 5)) {
        if (!job.job_id) continue
        try {
          const s = await api(`/api/training/status?provider=${job.provider}&job_id=${encodeURIComponent(job.job_id)}`)
          setStatuses(prev => ({ ...prev, [job.job_id]: s }))
          const m = s?.data?.metrics as Record<string, unknown> | undefined
          if (m) setTraces(prev => ({
            ...prev,
            [job.job_id]: pushLoss(prev[job.job_id] ?? [], m.latest_step, m.latest_loss),
          }))
        } catch { /* transient */ }
      }
    }, 5000)
    return () => clearInterval(id)
  }, [jobs])

  const set = (k: string, v: string) => setForm(f => ({ ...f, [k]: v }))

  const submit = async (validateOnly: boolean) => {
    setBusy(true); setMsg(null)
    const body = {
      provider: form.provider,
      dataset_root: form.dataset_root || undefined,
      base_model: form.base_model || undefined,
      output_dir: form.output_dir || undefined,
      steps: Number(form.steps) || 10000,
      method: form.method || undefined,
    }
    try {
      const j = await post(validateOnly ? '/api/training/validate' : '/api/training/submit', body)
      setMsg(j.status === 'success' ? `✓ ${j.text?.slice(0, 200)}` : `✗ ${j.text?.slice(0, 300)}`)
      if (!validateOnly && j.status === 'success') refresh()
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
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
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
    setBusy(false)
  }

  const replay = async (d: Dataset) => {
    setBusy(true); setMsg(null)
    try {
      const j = await post('/api/replay', { repo_id: d.repo_id, root: d.root, episode: 0 })
      setMsg(j.peer_id
        ? `▶ replaying ${d.repo_id} ep0 as ${j.peer_id} — watch it in the fleet grid`
        : `⚠ ${JSON.stringify(j).slice(0, 200)}`)
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
    setBusy(false)
  }

  const exportCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      setMsg(j.status === 'success' ? `✓ ${j.text?.slice(0, 250)}` : `✗ ${j.text?.slice(0, 250)}`)
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
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
    } catch (e) { setMsg(`⚠ ${(e as any)?.message ?? e}`) }
    setBusy(false)
  }


  // The form re-told as one sentence. A grid of fields answers "what can I
  // set"; the sentence answers the question that actually matters before a
  // multi-hour job: "what did I just ask for?"
  const datasetLabel = (() => {
    const d = datasets.find(x => x.root === form.dataset_root)
    return d ? `${d.repo_id} (${d.total_episodes ?? '?'} eps)` : null
  })()
  const story = form.dataset_root
    ? `Fine-tune ${form.base_model || 'lerobot/smolvla_base'} on ${datasetLabel ?? form.dataset_root} for ${fmtStep(Number(form.steps) || 10000)} steps (${form.method}), saving to ${form.output_dir || '…pick an output dir'}.`
    : 'Pick a dataset to begin — the plan reads back here before anything runs.'

  return (
    <div className="train-sheet">
      <div className="train-head">
        <h2>🎓 Training</h2>
        <button className="dock-min" onClick={onClose}>✕</button>
      </div>

      <div className="train-form">
        <p className={`train-story${form.dataset_root ? '' : ' empty'}`}>{story}</p>
        <label className="field"><span>provider</span>
          <select value={form.provider} onChange={e => set('provider', e.target.value)} disabled={busy}>
            {trainers.map(t => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label className="field"><span>dataset</span>
          <select value={form.dataset_root} onChange={e => set('dataset_root', e.target.value)} disabled={busy}>
            <option value="">— pick a local dataset —</option>
            {datasets.map(d => (
              <option key={d.root} value={d.root}>
                {d.repo_id} ({d.total_episodes ?? '?'} eps{d.robot_type && d.robot_type !== 'unknown' ? `, ${d.robot_type}` : ''})
              </option>
            ))}
          </select>
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
          <button className="btn go wide" onClick={() => submit(false)} disabled={busy || !form.dataset_root || !form.output_dir}>▶ train</button>
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
        {datasets.length === 0 && <div className="dock-hint">No local LeRobotDatasets found.</div>}
        {datasets.map(d => (
          <div className="train-job" key={d.root}>
            <div className="train-job-head">
              <b>{d.repo_id}</b>
              <span className="jstate">{d.total_episodes ?? '?'} eps · {d.fps ?? '?'} fps</span>
            </div>
            <div className="train-job-actions">
              <button className="btn ghost" onClick={() => replay(d)} disabled={busy}
                title="Replay episode 0 in a live mesh sim — appears in the fleet grid">
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
          return (
            <div className="train-job" key={job.job_id ?? Math.random()}>
              <div className="train-job-head">
                <b>{job.provider}</b>
                <span className={`jstate ${state}`}>{state}</span>
              </div>
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
