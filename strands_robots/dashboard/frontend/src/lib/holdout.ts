/**
 * val_episodes: hold out the LAST N episodes as a validation set.
 *
 * Why this field exists at all: without it every run this screen starts trains on 100% of the
 * episodes and reports a TRAINING loss only — a curve that falls just as prettily when a policy
 * memorises the dataset as when it learns the task. It was missing from the form's vocabulary
 * entirely (SPEC_KEYS), so asking for a holdout came back as "unknown field(s): val_episodes".
 *
 * Empty is a real answer, not an unfilled field: it means "train on every episode", which is what
 * the backend does with `null`. So this helper never invents a default — the only thing it refuses
 * is a value that would silently do something other than what it looks like:
 *
 *  - 0 or negative: the backend converts the count into a split fraction, and a non-positive value
 *    produces NO SPLIT AT ALL. Accepting it would show a holdout in the form and train without one.
 *  - fractional: lerobot takes the ceiling of that fraction, so 2.7 reserves 3, not 2. A number
 *    that means something else than it reads is refused rather than quietly rounded.
 *  - >= the dataset's episode count: nothing (or nearly nothing) is left to train on. The trainer
 *    also refuses this — it reads meta/info.json — but the dataset picker already knows the count,
 *    so saying it here costs no round trip. When the count is unknown (a Hub source with no local
 *    root), this helper stays quiet and lets the trainer be the one to refuse: guessing a bound
 *    would block a legitimate run.
 */

export interface Holdout {
  /** what to put in the submit body, or null to omit the key entirely */
  send: number | null
  /** why this value cannot be sent, or null */
  problem: string | null
  /** the sentence under the input: what WILL happen with this value */
  say: string
}

const NO_SPLIT =
  'empty: trains on every episode and reports training loss only — a validation split is what tells learning from memorising'

export function holdout(raw: string, episodeCount?: number | null): Holdout {
  const text = (raw ?? '').trim()
  if (!text) return { send: null, problem: null, say: NO_SPLIT }

  const n = Number(text)
  if (!Number.isFinite(n)) {
    return { send: null, problem: `“${text}” is not a number`, say: NO_SPLIT }
  }
  if (n <= 0) {
    return {
      send: null,
      problem: `a holdout of ${n} reserves no episodes at all — leave it empty to train on every episode`,
      say: NO_SPLIT,
    }
  }
  if (!Number.isInteger(n)) {
    return {
      send: null,
      problem: `${text} would reserve ${Math.ceil(n)} episodes, not ${Math.floor(n)} — enter a whole number`,
      say: NO_SPLIT,
    }
  }

  const known = typeof episodeCount === 'number' && Number.isFinite(episodeCount) && episodeCount > 0
  if (known && n >= (episodeCount as number)) {
    return {
      send: null,
      problem: `this dataset has ${episodeCount} episodes — holding out ${n} leaves ${Math.max(0, (episodeCount as number) - n)} to train on`,
      say: NO_SPLIT,
    }
  }

  const rest = known ? `${(episodeCount as number) - n} to train on, ` : ''
  return {
    send: n,
    problem: null,
    say: `holds out the last ${n} episode${n === 1 ? '' : 's'} for validation (${rest}so an eval loss is logged alongside the training loss)`,
  }
}
