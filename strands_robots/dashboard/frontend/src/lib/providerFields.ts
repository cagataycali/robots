/**
 * Q49: which extra field a training provider needs, and what to say about it.
 *
 * GR00T's validate() refuses with "embodiment is required for GR00T (--embodiment_tag)" — a real
 * requirement, and `embodiment` is a real spec key, so the honest answer is not to refuse the
 * provider (that is Q48's case, for providers the form CANNOT express) but to grow the field the
 * moment it becomes relevant. Showing it for every provider would be worse: an input that does
 * nothing for lerobot_local teaches people to ignore inputs.
 *
 * This mirrors a requirement the SDK declares only inside a trainer's validate(), so
 * tests/test_dashboard_trainer_form_support.py asks validate whether the mirror still holds.
 */

export interface ExtraField {
  /** the spec key, sent verbatim in the submit body */
  key: 'embodiment'
  label: string
  placeholder: string
  /** shown under the input: what it is for, in the words the trainer uses */
  say: string
  /** true when validate() refuses without it — the form can then say so before submitting */
  required: boolean
}

const EMBODIMENT: ExtraField = {
  key: 'embodiment',
  label: 'embodiment',
  placeholder: 'new_embodiment',
  say: 'GR00T needs an embodiment tag (--embodiment_tag): which robot body the data came from',
  required: true,
}

/** The extra fields the chosen provider needs. Unknown providers get none. */
export function extraFields(provider: string): ExtraField[] {
  return provider === 'groot' ? [EMBODIMENT] : []
}

/**
 * What is still missing for this provider, in one operator sentence — or '' when nothing is.
 * The form shows this BEFORE the submit button, so a refusal that is certain is not delivered
 * as a surprise by the server.
 */
export function missingForProvider(provider: string, values: Record<string, string>): string {
  const missing = extraFields(provider).filter(f => f.required && !(values[f.key] || '').trim())
  if (!missing.length) return ''
  return missing.map(f => f.say).join(' · ')
}
