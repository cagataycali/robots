/**
 * What a screen reader hears when the fleet agent answers (Q157).
 *
 * The dock's transcript had NO semantics at all: no role, no name, no live region. So a
 * sighted operator watched the reply arrive and a screen-reader user heard nothing —
 * they typed into a box and the app went silent, with no way to know a reply existed
 * short of hunting for it.
 *
 * The obvious fix is wrong. AgentDock.patchAgent APPENDS DELTAS to the trailing bubble,
 * so the transcript mutates once per token: as a live region it would stutter the answer
 * word by word, re-interrupting itself for the length of the reply. That is worse than
 * silence, because it is unstoppable and unreadable. The transcript is therefore a `log`
 * with live updates explicitly OFF, and this rule composes ONE sentence to announce when
 * the turn is finished — the same "narrow the claim, announce the settled fact" move the
 * rest of this dashboard makes.
 *
 * Announcing nothing while busy is deliberate: a spinner is not news, and the reply that
 * follows is the news.
 */
export interface AnnounceMsg {
  role: string
  text: string
  tools?: { name: string; status: string }[]
  delivered?: boolean | null
}

export interface AnnounceInput {
  /** true while the reply is still streaming in */
  busy: boolean
  /** the last entry in the transcript, whatever kind it is */
  last?: AnnounceMsg | null
  /** a socket-level failure, which outranks the transcript */
  error?: string | null
}

/** Unbounded speech is its own trap: a 4000-character answer read aloud cannot be skipped
 *  in every reader, so a long reply is announced up to here and then points at the page. */
export const SPOKEN_MAX = 600

export function clip(text: string, max: number = SPOKEN_MAX): string {
  const t = text.trim()
  if (t.length <= max) return t
  const cut = t.slice(0, max)
  const at = cut.lastIndexOf(' ')
  // Cut at a word so the announcement does not end mid-word, unless there is no space.
  return `${(at > max * 0.6 ? cut.slice(0, at) : cut).trimEnd()}… the rest is in the conversation`
}

export function turnAnnouncement({ busy, last, error }: AnnounceInput): string {
  // A refused socket is about the whole dock, not about one message, and it is the only
  // thing worth interrupting a stream for.
  if (error) return `the fleet agent could not be reached: ${error}`
  // Silence while the answer is still arriving — see the module note.
  if (busy) return ''
  if (!last) return ''
  if (last.role === 'notice') return `notice from the fleet: ${clip(last.text)}`
  // A user bubble left as the last entry means the send failed: the agent never spoke.
  if (last.role === 'user') {
    return last.delivered === false
      ? 'your message was not delivered — use "send again" to retry it'
      : ''
  }
  if (last.role !== 'agent') return ''
  const ran = (last.tools ?? []).length
  const body = last.text.trim()
  // Tools with no words is a real outcome (an action taken and nothing said), and it must
  // not be announced as an empty reply.
  if (!body) return ran ? `the agent ran ${ran === 1 ? '1 tool' : `${ran} tools`} and said nothing` : ''
  const prefix = ran ? `the agent replied after ${ran === 1 ? '1 tool' : `${ran} tools`}: ` : 'the agent replied: '
  return prefix + clip(body)
}
