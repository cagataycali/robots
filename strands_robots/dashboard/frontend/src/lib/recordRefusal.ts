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

/**
 * The extra request fields for a retry, given what the operator ticked.
 *
 * The flag is sent ONLY when the operator ticked the box in front of the
 * refusal that named it. It is never a default and never remembered: a new
 * refusal must be read and answered again, because "yes" to a stale camera is
 * not "yes" to a camera that changed identity.
 */
export function overrideBody(
  offered: RecordOverride | null,
  acknowledged: boolean,
): Record<string, true> {
  if (!offered || !acknowledged) return {}
  return { [offered.flag]: true }
}
