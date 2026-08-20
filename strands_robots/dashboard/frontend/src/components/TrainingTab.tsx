import { useEffect, useRef, useState } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import { numField } from '../lib/numField'
import { trainingFreshness } from '../lib/trainingFreshness'
import { api, post, HttpError } from '../lib/endpoints'
import { extraFields, missingForProvider } from '../lib/providerFields'
import { sideEffectVerdict, type SideEffectKind } from '../lib/submitOutcome'
import LossSpark from './LossSpark'
import { pushLoss, fmtStep, type LossPoint } from '../lib/lossTrace'
import { setDeployIntent } from '../lib/deployIntent'
import { datasetKey as dsKey, selectDataset, selectionKey, replayable, trainable, selectedRow, datasetMark, type DatasetRow } from '../lib/datasetSelection'
import { datasetHint, isCurrentResponse } from '../lib/datasetHint'
import { jobsLedgerNotice } from '../lib/jobsLedger'
import { orderJobsNewestFirst } from '../lib/orderJobs'
import { newerThanApplied } from '../lib/requestOrder'
import { outputDirSay, trainGate, type OutputDirVerdict } from '../lib/outputDirIntent'

// One row is either LOCAL (has a `root` path, trains offline) or from the HUB
// (no root, trains from repo_id after a download). lib/datasetSelection owns
// that distinction and the rule that exactly one field reaches the trainer.
type Dataset = DatasetRow
interface Job { job_id: string; provider: string; dataset?: string; base_model?: string; output_dir?: string; steps?: number; submitted_at?: number }
interface JobStatus { status: string; data: { status?: string; metrics?: Record<string, unknown> }; text: string }
/**
 * What the server could see of the exported artifact ON DISK (Q36). `ok` means "nothing
 * objectionable there", NOT "this policy loads" - the server deliberately never loads the
 * model (seconds to minutes, and an OOM risk on a box that is mid-training), so nothing here
 * may be rendered as a guarantee. Absent = an older server: then the old behaviour stands,
 * because a missing verdict is not a bad verdict.
 */
interface ArtifactVerdict { ok?: boolean; reason?: string; message?: string; warning?: string; path?: string }

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
  /**
   * Q48: providers this FORM cannot submit, mapped to why. `ppo` and `fast_sac` need an
   * RLTrainSpec built in a script — picking one here spent a dataset choice and a click to
   * earn "ppo requires an RLTrainSpec, got TrainSpec", a sentence about internal classes on
   * a path that could never succeed. Server-derived, never guessed here: a provider the
   * backend cannot classify stays fully selectable.
   */
  const [unsupported, setUnsupported] = useState<Record<string, string>>({})
  /* Q58: focus must land inside an overlay and go back to whatever opened it. */
  const sheetRef = useRef<HTMLDivElement | null>(null)
  useDialogFocus(sheetRef)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [statuses, setStatuses] = useState<Record<string, JobStatus>>({})
  // WHEN each status was last read, and how many reads have failed since - a
  // swallowed poll error used to leave a dead run rendering as healthy progress.
  const [polledAt, setPolledAt] = useState<Record<string, number>>({})
  const [pollFail, setPollFail] = useState<Record<string, { n: number; msg: string }>>({})
  const [nowS, setNowS] = useState(() => Date.now() / 1000)
  const [traces, setTraces] = useState<Record<string, LossPoint[]>>({})
  const [form, setForm] = useState({ provider: 'lerobot_local', dataset_root: '', dataset_repo_id: '', base_model: 'lerobot/smolvla_base', output_dir: '', steps: '10000', method: 'lora', embodiment: '' })
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
  // The status poll's own ordering: `tick` counts polling rounds, `applied` is the
  // round whose answer is on screen for each job. A round is skipped while the
  // previous one is still in flight - see lib/requestOrder.ts for why a late
  // answer is worse than no answer here (it wipes the loss curve).
  const tick = useRef(0)
  const tickBusy = useRef(false)
  const applied = useRef<Record<string, number>>({})
  const [dsAuth, setDsAuth] = useState<{ authenticated?: boolean; user?: string | null; detail?: string } | null>(null)
  // The ledger is a file that can go unreadable; an empty list alone cannot say so.
  const [jobsProblem, setJobsProblem] = useState<string | null>(null)
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
      setUnsupported(t.unsupported ?? {})
      // Newest first from the DATA, not from the file's shape: the next effect polls
      // only the first five, so an out-of-order ledger used to give status to the
      // finished runs and none to the one just started.
      setJobs(orderJobsNewestFirst(j.jobs ?? []))
      setJobsProblem(j.problem ?? null)
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

  // Q58: ask what is in the output dir while the operator types. Read-only endpoint; a failure
  // leaves the verdict null, which arms nothing and blocks nothing (the backend still refuses a
  // destructive launch, so a probe outage costs a clearer message, never a delete).
  const outSeq = useRef(0)
  useEffect(() => {
    const path = form.output_dir.trim()
    setClearArmedFor(null)  // editing the field revokes a tick made for the old path
    if (!path) { setOutDir(null); return }
    const seq = ++outSeq.current
    const t = setTimeout(async () => {
      try {
        const v = await api(`/api/training/output-dir?path=${encodeURIComponent(path)}`)
        if (!isCurrentResponse(seq, outSeq.current)) return
        setOutDir(v)
      } catch {
        if (!isCurrentResponse(seq, outSeq.current)) return
        setOutDir(null)
      }
    }, 400)
    return () => clearTimeout(t)
  }, [form.output_dir])


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
      // A round that takes longer than the interval used to have the next round
      // start on top of it: overlapping rounds pile up into a request storm
      // against a provider that is already slow, and their answers arrive out of
      // order. One round at a time; a skipped tick costs 5s of freshness, which
      // the age readout reports honestly.
      if (tickBusy.current) return
      tickBusy.current = true
      const round = ++tick.current
      try {
      // the five NEWEST (orderJobsNewestFirst put them there) - a finished run's
      // status cannot change, and the poll budget belongs to the live ones
      for (const job of jobs.slice(0, 5)) {
        if (!job.job_id) continue
        try {
          const s = await api(`/api/training/status?provider=${job.provider}&job_id=${encodeURIComponent(job.job_id)}`)
          // Only a newer round may speak for this job. An older answer landing
          // late is not merely stale: pushLoss reads its lower step as a RESTART
          // and drops the whole curve, while polledAt would stamp it as fresh.
          if (!newerThanApplied(round, applied.current[job.job_id])) continue
          applied.current[job.job_id] = round
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
      } finally { tickBusy.current = false }
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
    // Q37: a dataset the server could not confirm ANY episodes in. Training on it fails after
    // the environment setup, the base-model download and the dataset scan - minutes of work and
    // a job in the ledger that has to be read to be understood. Refused HERE, before the request,
    // and continuable: the check reads metadata only, so an operator who knows better (a dataset
    // written by something else, metadata about to be rebuilt) can insist once.
    const picked = selectedRow(datasets, form)
    const can = trainable(picked)
    if (!can.ok) {
      if (dsOverride !== selectionKey(form)) {
        setDsWarn({ key: selectionKey(form), reason: can.reason, recording: picked?.recording === true })
        setMsg(null)
        return
      }
      // CONSUMED BY THE RUN IT AUTHORISED. Found by scripts/audit-dataset-abandoned-hold.mjs:
      // a sticky override meant the FIRST insistence silenced the check for that dataset for the
      // rest of the session, so a second job hours later - by which time the operator may have
      // deleted the folder, or a recording may have half-filled it - started on a stale decision
      // nobody re-made. Same rule as deployIntent: one click authorises one action.
      setDsOverride(null)
    }
    // Q58: a run whose output_dir cannot be used - or whose contents would be DELETED without a
    // deliberate tick - never leaves the browser. validate() writes nothing, so it is exempt.
    if (!validateOnly && !gate.ok) { setMsg(`✗ ${gate.why}`); return }
    setBusy(true); setMsg(null)
    const body = {
      provider: form.provider,
      dataset_root: form.dataset_root || undefined,
      dataset_repo_id: form.dataset_repo_id || undefined,
      base_model: form.base_model || undefined,
      output_dir: form.output_dir || undefined,
      // Q58: consent to deleting what is already in that directory, carried only when the
      // operator ticked the box FOR THIS PATH. A run that needs it and lacks it never leaves
      // the browser (the gate below refuses first), so this is never a silent yes.
      ...(gate.confirmClear ? { confirm_clear: true } : {}),
      steps: wantedSteps.value,
      method: form.method || undefined,
      // Q49: sent only when the chosen provider asks for it. An empty string would reach
      // train_policy as a real value and GR00T would tag the dataset with "".
      ...(extraFields(form.provider).some(f => f.key === 'embodiment') && form.embodiment.trim()
        ? { embodiment: form.embodiment.trim() } : {}),
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

  /** Q37: the picked dataset's refusal, and the one key the operator has insisted on. */
  const [dsWarn, setDsWarn] = useState<{ key: string; reason: string; recording?: boolean } | null>(null)
  const [dsOverride, setDsOverride] = useState<string | null>(null)
  // Q58: what a run would DO to the typed output_dir, and the path the operator has ticked
  // for. `clearArmedFor` holds a PATH, not a boolean, so a yes given for one directory can
  // never delete another after the field is edited (same rule as dsOverride/deployIntent).
  const [outDir, setOutDir] = useState<OutputDirVerdict | null>(null)
  const [clearArmedFor, setClearArmedFor] = useState<string | null>(null)

  // Q58: the verdict's wording and whether the run may start at all.
  const outSay = outputDirSay(outDir)
  const gate = trainGate({ path: form.output_dir, verdict: outDir, armedFor: clearArmedFor })


  const [collect, setCollect] = useState({ dataset_root: '', instruction: 'pick up the red cube', n_episodes: '5', duration: '10', robot_name: 'so101' })
  const [showCollect, setShowCollect] = useState(false)

  /* Q60: `type="number"` hands you "" for junk and `Number(raw) || 10000` reads that as consent to
     a 10k-step run; `||` also lets a minus sign through, so `steps: -100` and `n_episodes: -3` were
     posted verbatim. Bounds stated, refusals explained, nothing corrected behind the operator. */
  const STEP_RULES = { what: 'steps', min: 1, max: 2_000_000, remedy: 'submit a shorter run' }
  const wantedSteps = numField(form.steps, STEP_RULES)
  const wantedEpisodes = numField(collect.n_episodes, { what: 'episodes', min: 1, max: 500, remedy: 'collect in batches' })
  const wantedSeconds = numField(collect.duration, { what: 'seconds per episode', min: 1, max: 600 })

  const submitCollect = async () => {
    if (!collect.dataset_root.trim()) return
    setBusy(true); setMsg(null)
    try {
      const j = await post('/api/collect', {
        dataset_root: collect.dataset_root,
        instruction: collect.instruction,
        n_episodes: wantedEpisodes.value,
        duration: wantedSeconds.value,
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

  /**
   * A checkpoint the server could not confirm on disk, held back from staging until the
   * operator says they mean it. A REFUSAL WITH A DOOR, like every other gate here: the disk
   * check can be wrong (a loader that infers its own weights layout, an artifact the server
   * cannot see through a symlink), and a hard block would leave the operator with a trained
   * run and no way to use it.
   */
  const [stageAnyway, setStageAnyway] = useState<{ job: Job; ckpt: string; message: string } | null>(null)

  const exportCkpt = async (job: Job) => {
    setBusy(true)
    try {
      const j = await post('/api/training/export', { provider: job.provider, output_dir: job.output_dir, dataset_root: job.dataset, base_model: job.base_model })
      const art: ArtifactVerdict | undefined = j?.artifact
      if (j.status !== 'success') setMsg(`✗ ${j.text?.slice(0, 250)}`)
      else if (j.deployable === false) {
        // The export RAN - that is why the trainer's ✓ is still shown - and what it produced
        // is not a policy. Said here, on the export button, because this is where an operator
        // looks before they go anywhere near a robot.
        setMsg(`⚠ the export succeeded but the artifact is not usable: ${art?.message ?? 'the checkpoint could not be confirmed on disk'}`)
      } else setMsg(`✓ ${j.text?.slice(0, 250)}${art?.warning ? ` — ⚠ ${art.warning}` : ''}`)
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
      const art: ArtifactVerdict | undefined = j?.artifact
      if (j.status !== 'success' || typeof ckpt !== 'string' || !ckpt) {
        setMsg(`✗ nothing deployable: ${j.text?.slice(0, 200) ?? 'export returned no artifact path'}`)
      } else if (j.deployable === false) {
        // Do NOT stage it. Staging prefills a run form the operator then presses Run on, and
        // by that point the checkpoint's problem has become "the arm did not move and I do not
        // know why". Hold it here, with the reason and a door.
        setStageAnyway({ job, ckpt, message: art?.message ?? 'the checkpoint could not be confirmed on disk' })
        setMsg(null)
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


  /** The operator overrode the disk check. Stage exactly what deploy would have staged. */
  const stageRegardless = () => {
    if (!stageAnyway) return
    const { job, ckpt } = stageAnyway
    setDeployIntent({
      checkpoint: ckpt,
      policy_type: guessPolicyType(job.base_model),
      source: `training job ${job.job_id} (${job.base_model || job.provider}) — staged over an unconfirmed artifact`,
    })
    setStageAnyway(null)
    setMsg('🚀 checkpoint staged over the warning — open a robot\u2019s run form; nothing runs until you press Run there')
  }

  // The form re-told as one sentence. A grid of fields answers "what can I
  // set"; the sentence answers the question that actually matters before a
  // multi-hour job: "what did I just ask for?"
  const datasetPicked = form.dataset_root || form.dataset_repo_id
  const datasetLabel = selectDataset(datasets, selectionKey(form)).label || null
  // The plan sentence reads back what WILL run, so with an unusable step count it must say that
  // rather than quietly print the 0 the parse produced — it used to print 10,000, the old fallback.
  const stepsPhrase = wantedSteps.problem ? `an unset number of` : fmtStep(wantedSteps.value)
  // Q49: a certain refusal belongs in the read-back, not in the response to a click. The plan
  // sentence is what people check before a multi-hour job, so if the provider will refuse for a
  // missing field it says so here too, next to the promise.
  const missingExtra = missingForProvider(form.provider, form as Record<string, string>)
  const story = datasetPicked
    ? `Fine-tune ${form.base_model || 'lerobot/smolvla_base'} on ${datasetLabel ?? datasetPicked} for ${stepsPhrase} steps (${form.method}), saving to ${form.output_dir || '…pick an output dir'}.`
      + (missingExtra ? ` This will be refused until you fill it in: ${missingExtra}.` : '')
    : 'Pick a dataset to begin — the plan reads back here before anything runs.'

  return (
    /* role + label like RecordPanel's sheet: this is a full-bleed layer over the fleet, and a
       screen reader that is not told it entered a dialog reads it as more of the page it just
       left. Same reason its ✕ is the only way out on a phone. */
    <div ref={sheetRef} className="train-sheet" role="dialog" aria-label="Training">
      <div className="train-head">
        <h2>🎓 Training</h2>
        <button className="dock-min" onClick={onClose} aria-label="close training" title="Escape">✕</button>
      </div>

      <div className="train-form">
        <p className={`train-story${datasetPicked ? '' : ' empty'}`}>{story}</p>
        <label className="field"><span>provider</span>
          <select value={form.provider} onChange={e => set('provider', e.target.value)} disabled={busy}>
            {/* Still LISTED, so the capability is not hidden and nobody hunts for a
                provider they know exists — but not selectable, with the reason attached
                rather than delivered as an error after the fact. */}
            {trainers.map(t => (
              <option key={t} disabled={t in unsupported}
                      title={unsupported[t] ?? undefined}>
                {t}{t in unsupported ? ' — not from this form' : ''}
              </option>
            ))}
          </select>
        </label>
        {/* One line per DISTINCT reason, names grouped: two providers refused for the same
            reason must not print the same long sentence twice, and two refused for different
            reasons must not be merged under one. */}
        {[...new Set(Object.values(unsupported))].map(reason => (
          <p className="hint" key={reason}>
            {Object.keys(unsupported).filter(k => unsupported[k] === reason).sort().join(' and ')}
            {' cannot be trained from here: '}{reason}.
          </p>
        ))}
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
                    {/* Q37: an abandoned recording's folder lists as a dataset for ever. Marked
                        IN THE OPTION, because the picker is where the choice is made - a warning
                        that only appears after selecting arrives one decision too late. */}
                    {datasetMark(d).glyph}{d.repo_id} ({d.total_episodes ?? '?'} eps{d.robot_type && d.robot_type !== 'unknown' ? `, ${d.robot_type}` : ''})
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
          <input value={form.output_dir} onChange={e => set('output_dir', e.target.value)} disabled={busy}
                 placeholder="/tmp/my_policy_ckpt" aria-describedby="train-outdir-say"
                 aria-invalid={outSay.blocked || outSay.confirmable} />
          {/* Q58: a run into an existing directory DELETES it (the trainer rmtree's a dir with no
              resumable checkpoint). This line is the only warning that exists before the loss, so
              it names the count and the files rather than saying "not empty". */}
          <span id="train-outdir-say" className={`fieldsay${outSay.tone === 'info' ? '' : ' bad'}`} role="status" aria-live="polite">
            {outSay.text ?? ''}
          </span>
        </label>
        {outSay.confirmable && (
          <label className="field check consent-clear">
            <input type="checkbox" checked={clearArmedFor === (outDir?.path ?? form.output_dir.trim())}
                   onChange={e => setClearArmedFor(e.target.checked ? (outDir?.path ?? form.output_dir.trim()) : null)}
                   disabled={busy} />
            <span>{outSay.confirmLabel}</span>
          </label>
        )}
        {/* Q49: appears only for the provider that needs it. GR00T's validate() refuses without
            an embodiment tag; before this the form had no field for it at all, so `groot` was a
            selectable option that could not be submitted whatever you typed. */}
        {extraFields(form.provider).map(f => (
          <label className="field" key={f.key}><span>{f.label}</span>
            <input value={form[f.key]} onChange={e => set(f.key, e.target.value)} disabled={busy}
                   placeholder={f.placeholder} aria-describedby={`train-${f.key}-say`}
                   aria-invalid={f.required && !form[f.key].trim()} />
            <span id={`train-${f.key}-say`} className={`fieldsay${f.required && !form[f.key].trim() ? ' bad' : ''}`}>
              {f.say}
            </span>
          </label>
        ))}
        <div className="train-row">
          <label className="field"><span>steps</span>
            <input type="number" value={form.steps} onChange={e => set('steps', e.target.value)} disabled={busy}
                   aria-invalid={!!wantedSteps.problem} aria-describedby="train-steps-say" />
            <span id="train-steps-say" className={`fieldsay${wantedSteps.problem ? ' bad' : ''}`}>
              {wantedSteps.problem ?? wantedSteps.note ?? ''}
            </span>
          </label>
          <label className="field"><span>method</span>
            <select value={form.method} onChange={e => set('method', e.target.value)} disabled={busy}>
              <option value="lora">lora</option>
              <option value="full">full</option>
            </select>
          </label>
        </div>
        <div className="train-actions">
          <button className="btn ghost" onClick={() => submit(true)} disabled={busy || !!wantedSteps.problem}>✓ validate</button>
          <button className="btn go wide" onClick={() => submit(false)}
                  disabled={busy || !datasetPicked || !!wantedSteps.problem || !gate.ok}
                  title={gate.why ?? undefined}>
            {outSay.confirmable && gate.ok ? '▶ delete and train' : '▶ train'}
          </button>
        </div>
        {msg && <div className="train-msg">{msg}</div>}
        {/* Not started, and not blocked either. The reason is the server's own sentence, which
            names the physical event (a session that opened and recorded nothing, frames that
            never landed) rather than calling the dataset invalid. */}
        {dsWarn && dsWarn.key === selectionKey(form) && (
          <div className="train-msg warn artifact-hold" role="alert">
            <div>{dsWarn.recording ? '⏺' : '⚠'} not started: {dsWarn.reason}</div>
            {/* Q38: for a LIVE session the useful next move is the record screen, not another
                dataset — and this tab cannot navigate, so it says where to look rather than
                pretending to take them there. */}
            {dsWarn.recording && <div className="jstate">the record screen shows this session's progress; training can start the moment it closes</div>}
            <div className="artifact-hold-actions">
              <button className="btn ghost" onClick={() => { setDsWarn(null); setDsOverride(null) }}>pick another dataset</button>
              <button className="btn" onClick={() => { setDsOverride(dsWarn.key); setDsWarn(null); setMsg('⚠ dataset warning overridden — press start training again') }}
                      title={dsWarn.recording
                        ? 'the trainer would read episodes as they are still being written - only useful if the session is about to close'
                        : 'the check reads metadata only - insist if you know the episodes are there'}>
                train on it anyway
              </button>
            </div>
          </div>
        )}
        {/* Held back, not blocked. The reason names the physical event (a config with no
            weights beside it, an unmounted volume) so the operator can decide whether they
            know better than the check - and the button says what they are overriding. */}
        {stageAnyway && (
          <div className="train-msg warn artifact-hold" role="alert">
            <div>⚠ not staged: {stageAnyway.message}</div>
            <div className="artifact-hold-actions">
              <button className="btn ghost" onClick={() => setStageAnyway(null)}>keep it unstaged</button>
              <button className="btn" onClick={stageRegardless}
                      title="the disk check can be wrong - stage it and find out on the run form">
                stage it anyway
              </button>
            </div>
          </div>
        )}
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
                <input type="number" value={collect.n_episodes} aria-invalid={!!wantedEpisodes.problem}
                  aria-describedby="collect-episodes-say"
                  onChange={e => setCollect(c => ({ ...c, n_episodes: e.target.value }))} disabled={busy} />
                <span id="collect-episodes-say" className={`fieldsay${wantedEpisodes.problem ? ' bad' : ''}`}>
                  {wantedEpisodes.problem ?? wantedEpisodes.note ?? ''}
                </span>
              </label>
              <label className="field"><span>sec/episode</span>
                <input type="number" value={collect.duration} aria-invalid={!!wantedSeconds.problem}
                  aria-describedby="collect-seconds-say"
                  onChange={e => setCollect(c => ({ ...c, duration: e.target.value }))} disabled={busy} />
                <span id="collect-seconds-say" className={`fieldsay${wantedSeconds.problem ? ' bad' : ''}`}>
                  {wantedSeconds.problem ?? wantedSeconds.note ?? ''}
                </span>
              </label>
              <label className="field"><span>robot</span>
                <input value={collect.robot_name}
                  onChange={e => setCollect(c => ({ ...c, robot_name: e.target.value }))} disabled={busy} />
              </label>
            </div>
            <div className="train-actions">
              <button className="btn go wide" onClick={submitCollect}
                disabled={busy || !collect.dataset_root.trim() || !!wantedEpisodes.problem || !!wantedSeconds.problem}>
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
                {/* Q38: "0 eps" on a dataset being recorded right now reads as empty when it is
                    filling — the count is a snapshot of flushed metadata, not of the session. */}
                {d.recording ? '⏺ recording now' : d.root
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
        {/* Rendered for ANY count: a partial list is the dangerous case, because
            the cards that survived make it look complete. */}
        {(() => {
          const notice = jobsLedgerNotice({ count: jobs.length, problem: jobsProblem })
          return notice.text ? (
            <div className={notice.tone === 'warn' ? 'dock-hint warn' : 'dock-hint'}>{notice.text}</div>
          ) : null
        })()}
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
