/**
 * A refused recording the operator can actually answer, on the screen.
 *
 * `/api/record/open` has three camera gates, and each one is deliberately
 * CONTINUABLE: the sentence it refuses with names the flag that proceeds anyway
 * (`ignore_dead_cameras`, `ignore_missing_cameras`, `ignore_camera_identity`).
 * Until now only the first was reachable — RecordPanel asks that question itself
 * from a client-side freshness check — so a session refused for the other two
 * left the operator reading a paragraph containing the name of a flag they had
 * no way to send. A continuable refusal that can only be continued with curl is
 * a dead end wearing a helpful message.
 *
 * This module reads the server's own words and says which override is on offer.
 * It does NOT rewrite the refusal: the explanation stays the server's, written
 * once next to the check that knows why. What it adds is the one line a
 * checkbox needs — the admission the operator is making, in the first person,
 * so ticking it is a statement and not a shrug.
 *
 * The ALLOWLIST is the safety property here. A 409 body is a sentence, and
 * offering an override because a message merely mentions the word "ignore"
 * would let any future refusal (or an error quoting one) grow a bypass button
 * nobody designed. Only these three flags exist, only exact matches count, and
 * anything else renders as text with no tick at all.
 */

/** The overrides `/api/record/open` accepts, and what ticking one admits to. */
export interface RecordOverride {
  /** The request field to send. Exactly as the backend spells it. */
  flag: 'ignore_dead_cameras' | 'ignore_missing_cameras' | 'ignore_camera_identity'
  /** First-person, present tense: what the operator is claiming by ticking. */
  label: string
  /** What the dataset will carry if the operator is wrong — never hidden. */
  cost: string
}

const OVERRIDES: RecordOverride[] = [
  {
    flag: 'ignore_dead_cameras',
    label: 'I know this camera is stale — record without waiting for it',
    cost: 'the episodes may carry a frozen image, or none, for that view',
  },
  {
    flag: 'ignore_missing_cameras',
    label: 'record without the camera this machine cannot see',
    cost: 'that view is simply absent from every episode',
  },
  {
    flag: 'ignore_camera_identity',
    label: 'my cameras really are at these indices — record with them as they stand',
    cost: 'if the numbering did shift, every episode records the WRONG view while looking healthy',
  },
]

/**
 * The override a refusal is offering, or null when it offers none.
 *
 * @param message The 409 detail, exactly as the server wrote it.
 *
 * Silent by default, on purpose:
 * - a message naming NO known flag gets no tick (a network error, a 500, an
 *   unrelated 409 like "a recording session is already open");
 * - a message naming MORE THAN ONE gets no tick either. That is not
 *   indecision: two faults refused at once are two different admissions, and a
 *   single box would collect consent for the one the operator did not read.
 *   The next refusal after fixing one of them offers its own tick.
 */
export function overrideOffered(message: unknown): RecordOverride | null {
  if (typeof message !== 'string' || !message) return null
  const named = OVERRIDES.filter(o => message.includes(o.flag))
  return named.length === 1 ? named[0] : null
}

/** The three flags, in the order `/api/record/open` checks them: dead, then missing, then identity. */
export type RecordOverrideFlag = RecordOverride['flag']
const FLAGS: RecordOverrideFlag[] = OVERRIDES.map(o => o.flag)

/**
 * Every admission the operator has made about THIS attempt sequence, not just the last one.
 *
 * THE BUG THIS EXISTS FOR (Q98): the route checks the three gates in order and each one is skipped
 * only by its own flag, so an attempt carrying ONE flag is still refused by an earlier gate. Sending
 * only the flag from the most recent refusal therefore ping-ponged forever:
 *
 *   attempt 1 -> refused: missing camera        (tick)
 *   attempt 2 -> ignore_missing, refused: identity drift   (tick, and the first admission is dropped)
 *   attempt 3 -> ignore_identity, refused: missing camera again ... and around it goes.
 *
 * And that pair is not exotic, it is THE SAME PHYSICAL EVENT: unplugging one camera makes it missing
 * AND renumbers every index after it, which is identity drift. So the second most likely camera fault
 * on a real desk could not be continued from the screen at all, in the module written to end exactly
 * that dead end.
 *
 * Accumulating is safe for the reason a single tick was: every flag in here was named by a refusal the
 * operator READ and ticked in front of. What must never happen is a flag arriving any other way, so
 * this drops anything not on the allowlist, keeps the canonical order, and cannot grow a duplicate.
 * The caller clears the whole set when the arms or the dataset change, because an admission about
 * these cameras is an admission about THAT robot.
 */
export function nextAcknowledged(
  prev: readonly string[],
  offered: RecordOverride | null,
  acknowledged: boolean,
): RecordOverrideFlag[] {
  const kept = FLAGS.filter(f => prev.includes(f))
  if (!offered || !acknowledged || kept.includes(offered.flag)) return kept
  return FLAGS.filter(f => kept.includes(f) || f === offered.flag)
}

/** Those admissions as request fields. Nothing else can get in: the allowlist is the safety property. */
export function overrideBodyFlags(flags: readonly string[]): Record<string, true> {
  const body: Record<string, true> = {}
  for (const f of FLAGS) if (flags.includes(f)) body[f] = true
  return body
}
