/**
 * R6: turning a picked dataset row into the ONE field training should receive.
 *
 * The backend states the rule in its own refusal — "a data source
 * (dataset_root or dataset_repo_id) and output_dir are required" — and they are
 * alternatives, not synonyms:
 *
 *   - a LOCAL dataset has a `root`: a path on this machine, trains offline;
 *   - a HUB dataset has none: it trains from `dataset_repo_id` and is
 *     downloaded when the job starts.
 *
 * Sending both would leave the trainer to pick one for you, and the one it
 * picks decides whether a multi-hour job reads the episodes you just recorded
 * or a stranger's dataset with the same name. So this returns exactly one and
 * blanks the other, always.
 */

export interface DatasetRow {
  root?: string
  repo_id: string
  local?: boolean
  downloads?: number | null
  total_episodes?: number
  robot_type?: string
  fps?: number
  /**
   * Q37: what the server could see of this dataset WITHOUT loading it. `usable === false` means
   * the metadata itself says nothing was recorded, or the frames are not there. Absent means an
   * older server — and a MISSING verdict is not a bad verdict, so the old behaviour must stand.
   * Hub rows never carry one: nothing on this machine can inspect a dataset that is not here yet.
   */
  usable?: boolean
  reason?: string
  problem?: string
  /** Q38: a recorder is writing into this dataset at this moment. */
  recording?: boolean
}

export interface DatasetSelection {
  dataset_root: string
  dataset_repo_id: string
  /** true when the trainer will download it before the first step */
  fromHub: boolean
  /** what the plan sentence should call it, or '' when nothing is picked */
  label: string
}

const EMPTY: DatasetSelection = { dataset_root: '', dataset_repo_id: '', fromHub: false, label: '' }

/** Stable option key for either kind — a Hub row has no path to key on. */
export function datasetKey(d: DatasetRow): string {
  return d.root ?? `hub:${d.repo_id}`
}

/**
 * Resolve a <select> value back to the fields training accepts.
 * An unknown key clears the selection rather than guessing: a stale key means
 * the list changed under the operator (a keystroke re-ran the search), and
 * training the previous row would be the one outcome nobody asked for.
 */
export function selectDataset(rows: DatasetRow[], key: string): DatasetSelection {
  if (!key) return EMPTY
  const d = rows.find(r => datasetKey(r) === key)
  if (!d) return EMPTY
  if (d.root) {
    return {
      dataset_root: d.root,
      dataset_repo_id: '',
      fromHub: false,
      label: `${d.repo_id}${d.total_episodes ? ` (${d.total_episodes} eps)` : ''}`,
    }
  }
  return {
    dataset_root: '',
    dataset_repo_id: d.repo_id,
    fromHub: true,
    label: `${d.repo_id} — downloaded from the Hub at start`,
  }
}

/** The <select> value for the current form state (round-trips selectDataset). */
export function selectionKey(sel: { dataset_root: string; dataset_repo_id: string }): string {
  if (sel.dataset_root) return sel.dataset_root
  return sel.dataset_repo_id ? `hub:${sel.dataset_repo_id}` : ''
}

/**
 * Can this row be replayed here? Replay reads episode 0 off this disk, so a
 * Hub row cannot — and saying so before the click beats a failure raised deep
 * inside a dataset loader.
 */
export function replayable(d: DatasetRow): { ok: boolean; reason: string } {
  if (!d.root) return { ok: false, reason: 'on the Hub, not on this machine — training downloads it; replay needs it local' }
  // Q37: replay reads EPISODE 0. A dataset whose metadata says zero episodes has no episode 0,
  // so the click cannot do anything but fail deep inside a loader — and this row exists at all
  // only because meta/info.json is written when a recording OPENS. The server's sentence is used
  // verbatim: it knows which of the failure modes this is, and re-wording it here would let the
  // two disagree.
  if (d.usable === false) return { ok: false, reason: d.problem ?? 'this dataset has no episodes to replay' }
  return { ok: true, reason: 'Replay episode 0 in a live mesh sim — appears in the fleet grid' }
}

/**
 * Q38: a dataset a recorder is writing into RIGHT NOW carries `recording: true`, and it is a
 * different thing from a folder an abandoned session left behind — even though metadata alone
 * cannot tell them apart. Marking both with the same ⚠ would say "something is wrong here" about
 * the recording the operator is deliberately making, so the live one gets the recording glyph and
 * the "not yet" reading, while ⚠ keeps meaning "this will not work".
 */
export function datasetMark(d: DatasetRow): { glyph: string; kind: 'recording' | 'problem' | 'ok' } {
  if (d.recording) return { glyph: '⏺ ', kind: 'recording' }
  if (d.usable === false) return { glyph: '⚠ ', kind: 'problem' }
  return { glyph: '', kind: 'ok' }
}

/**
 * Q37: may training start on this row, and if not, why. Separate from `replayable` because the
 * two verbs fail differently — replay dies in seconds on episode 0, training dies after the
 * environment setup, the model download and the dataset scan it charged you for.
 */
export function trainable(d: DatasetRow | null): { ok: boolean; reason: string } {
  if (!d) return { ok: true, reason: '' }
  if (d.usable === false) return { ok: false, reason: d.problem ?? 'this dataset has no episodes to train on' }
  return { ok: true, reason: '' }
}

/** The picked row, or null when the form's selection is not in the current list. */
export function selectedRow(rows: DatasetRow[], sel: { dataset_root: string; dataset_repo_id: string }): DatasetRow | null {
  return rows.find(r => datasetKey(r) === selectionKey(sel)) ?? null
}
