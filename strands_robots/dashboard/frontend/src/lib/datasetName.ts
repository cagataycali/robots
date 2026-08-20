/**
 * Is the dataset name the operator is typing already taken? (Q39, part 2)
 *
 * The backend refuses a taken name before it parks the arms — but at that point the operator has
 * already picked a pair, aimed two cameras and pressed the button. The listing the training picker
 * uses answers this question, so the record form can say it while there is still nothing at stake.
 *
 * Deliberately NOT a validator: it never blocks the submit, and it never rewrites the field. It
 * reports what exists under that name and offers ONE free alternative to tap, because the operator
 * naming a dataset "so101-cubes" a second time usually wants "so101-cubes-2", not a lecture.
 */
export type KnownDataset = { repo_id?: string; root?: string; total_episodes?: number; local?: boolean }

/** Local rows only: a Hub dataset of the same name is not what a local recording would collide with. */
function localNames(known: KnownDataset[]): Map<string, number | undefined> {
  const out = new Map<string, number | undefined>()
  for (const d of known) {
    if (d.local === false) continue
    const id = (d.repo_id ?? '').trim()
    if (id) out.set(id, typeof d.total_episodes === 'number' ? d.total_episodes : undefined)
  }
  return out
}

/**
 * The next free name in the "-N" family: cubes -> cubes-2 -> cubes-3.
 *
 * A name already ending in -N counts as that N, so pressing the suggestion twice walks forward
 * instead of producing "cubes-2-2". The search is bounded: a fleet with hundreds of takes gets an
 * honest empty answer rather than a hang.
 */
export function freeVariant(name: string, known: KnownDataset[]): string | null {
  const taken = localNames(known)
  const base = name.trim()
  if (!base) return null
  const m = base.match(/^(.*?)-(\d+)$/)
  const stem = m ? m[1] : base
  let n = m ? Number(m[2]) : 1
  for (let i = 0; i < 200; i += 1) {
    n += 1
    const candidate = `${stem}-${n}`
    if (!taken.has(candidate)) return candidate
  }
  return null
}

/**
 * What to say about this name, or null to say nothing.
 *
 * `null` covers the two cases that must stay silent: a name nobody has used, and a name we have no
 * evidence about (the listing failed, or has not arrived). Silence here means "no reason to worry",
 * which is the honest reading — the backend still refuses on the way in, so this can only ever warn
 * EARLIER, never instead.
 */
export function nameVerdict(
  name: string,
  known: KnownDataset[] | null,
): { message: string; suggestion: string | null } | null {
  const id = (name ?? '').trim()
  if (!id || !known) return null
  const taken = localNames(known)
  if (!taken.has(id)) return null
  const episodes = taken.get(id)
  const what = typeof episodes === 'number' && episodes > 0
    // The number is the whole point: 40 episodes is an afternoon of hand-guiding, and the operator
    // is the only one who knows whether this is the same take or a name they forgot they used.
    ? `already exists with ${episodes} episode(s)`
    : 'already exists on disk (no episodes recorded — probably an interrupted session)'
  return {
    message: `“${id}” ${what}. Recording refuses to reuse a dataset directory, so this will be turned away when you press start.`,
    suggestion: freeVariant(id, known),
  }
}
