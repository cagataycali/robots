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
  /** frames actually written; 0 alongside 0 episodes is the abandoned-session shape. */
  total_frames?: number | null
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
 * Can this row be replayed here? Replay reads an episode off this disk (episode 0 unless the operator picks
 * another one — see episodeChoice below), so a
 * Hub row cannot — and saying so before the click beats a failure raised deep
 * inside a dataset loader.
 */
/**
 * MEASURED 2026-08-21, and the reason the two functions below no longer trust the verdict alone:
 * this machine's disk holds `local/sim_recording` — 0 episodes, 0 frames, an abandoned session's
 * leftovers — and the RUNNING dashboard predates `dataset_verdict`, so its rows arrive with no
 * `usable` field at all. Against that server the screen offered the empty folder as a normal
 * training target: the run would charge an environment setup and a model download before dying
 * inside a dataset loader, and the operator would read it as a broken trainer.
 *
 * `total_episodes` has been in every version of that response, so a zero there is evidence the page
 * already holds. Defence in depth for the case that is LIVE right now (a page newer than its
 * server), deliberately narrow:
 *   - only an explicit, finite 0 counts. undefined/null is no evidence and stays allowed.
 *   - the server's verdict WINS when present, because its sentence names which failure mode this is;
 *     this fallback only speaks when the field is missing entirely.
 */
function noEpisodes(d: DatasetRow): boolean {
  if (d.usable !== undefined) return false
  return d.total_episodes === 0 || d.total_frames === 0
}

/** The sentence for a row whose own count says it is empty — the server's wording, minus the part only it can know. */
const EMPTY_ROW = '0 episodes. meta/info.json is written when a recording session OPENS, before the first '
  + 'episode is captured, so a directory like this is what an abandoned session leaves behind — not a dataset. '
  + 'Record into it, or delete it.'

export function replayable(d: DatasetRow): { ok: boolean; reason: string } {
  if (!d.root) return { ok: false, reason: 'on the Hub, not on this machine — training downloads it; replay needs it local' }
  // Q37: replay reads an episode off the disk. A dataset whose metadata says zero episodes has none to read,
  // so the click cannot do anything but fail deep inside a loader — and this row exists at all
  // only because meta/info.json is written when a recording OPENS. The server's sentence is used
  // verbatim: it knows which of the failure modes this is, and re-wording it here would let the
  // two disagree.
  if (d.usable === false) return { ok: false, reason: d.problem ?? 'this dataset has no episodes to replay' }
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW }
  // Deliberately says no episode NUMBER: the number lives in the box next to the button now, and this
  // sentence is composed with episodeChoice's in the tooltip — two claims about the index would contradict.
  return { ok: true, reason: 'Replay in a live mesh sim — appears in the fleet grid' }
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
  // Same evidence, same glyph: a row the buttons will refuse must not look normal in the list.
  if (noEpisodes(d)) return { glyph: '⚠ ', kind: 'problem' }
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
  if (noEpisodes(d)) return { ok: false, reason: EMPTY_ROW }
  return { ok: true, reason: '' }
}

/** The picked row, or null when the form's selection is not in the current list. */
export function selectedRow(rows: DatasetRow[], sel: { dataset_root: string; dataset_repo_id: string }): DatasetRow | null {
  return rows.find(r => datasetKey(r) === selectionKey(sel)) ?? null
}

/**
 * Which episode a replay click should ask for — and whether the number the operator typed exists.
 *
 * Until now the replay button sent `episode: 0`, hardcoded, while the backend has always validated and
 * accepted any index. So a dataset of 40 episodes could only ever be watched at its first one, and the
 * operator had no way to see that: the button said "replay in sim", not "replay the first of 40". That is
 * a plausible reading of the report that replay "is not working properly" — a second recording attempt
 * replays the FIRST attempt, forever.
 *
 * Rules, in the order a human would apply them:
 *  - a blank box means episode 0, because that is the old behaviour and the least surprising default;
 *  - a non-integer or negative index is refused here rather than by the server, so the operator gets the
 *    sentence next to the box they typed in;
 *  - `total_episodes` is the server's own count: index >= N is refused NAMING N and the last valid index,
 *    which is the mistake this box will actually produce (off-by-one on a 1-based mental model);
 *  - an ABSENT count is not zero episodes. Older servers and Hub rows do not carry it, so the index is
 *    passed through unchecked and the caller is told the count is unknown — a missing fact must not
 *    become a refusal (the same law as `usable` above).
 */
export function episodeChoice(
  d: DatasetRow,
  requested?: number | string | null,
): { ok: boolean; episode: number; reason: string; countKnown: boolean } {
  const total = typeof d.total_episodes === 'number' && Number.isFinite(d.total_episodes) ? d.total_episodes : null
  const countKnown = total !== null && total > 0
  const raw = typeof requested === 'string' ? requested.trim() : requested
  if (raw === undefined || raw === null || raw === '') {
    return { ok: true, episode: 0, reason: countKnown ? `episode 0 of ${total}` : 'episode 0', countKnown }
  }
  const n = Number(raw)
  if (!Number.isInteger(n)) {
    return { ok: false, episode: 0, reason: `“${raw}” is not a whole episode number`, countKnown }
  }
  if (n < 0) return { ok: false, episode: 0, reason: 'episode numbers start at 0', countKnown }
  if (total !== null && total <= 0) {
    return { ok: false, episode: 0, reason: 'this dataset records no episodes yet', countKnown }
  }
  if (total !== null && n >= total) {
    return {
      ok: false,
      episode: 0,
      reason: `this dataset has ${total} episode${total === 1 ? '' : 's'}, numbered 0–${total - 1}`,
      countKnown,
    }
  }
  return {
    ok: true,
    episode: n,
    reason: countKnown ? `episode ${n} of ${total}` : `episode ${n} (this server does not report a count)`,
    countKnown,
  }
}
