/**
 * WHEN to ask the server whether a newer bundle exists.
 *
 * MEASURED 2026-08-20: cagatay's phone, open on the dashboard from Seattle, was
 * running an ELEVEN HOUR OLD bundle and opening 1.5 camera websockets a second
 * (69,653 in this server's lifetime, all for a camera that was publishing fine —
 * a local socket from the same machine survived 20s at 4.7fps, so the server was
 * not hanging up). Whatever that phone's bundle does, it is not what the current
 * code does, and no fix we shipped that day could reach it.
 *
 * The cause is structural, not a bug in the update PROMPT: a service worker checks
 * for a new build when it REGISTERS — i.e. on page load. A dashboard is a cockpit
 * left open for days on a phone beside the arms, so registration happens once and
 * the check never happens again. "The operator decides when to reload" (see
 * usePwa: an auto-reload would tear down camera sockets and the run form of a
 * moving robot) is only a real decision if they are ever ASKED.
 *
 * So: poll, but on terms that suit a phone on cellular — not while offline (a
 * failed check is noise and costs battery), not while hidden (a backgrounded tab
 * gets throttled and would just pile up timers), and never faster than the
 * interval, no matter how often the page is brought back to the foreground.
 */

/** How often a page left open should ask again. */
export const SW_UPDATE_INTERVAL_MS = 15 * 60 * 1000

export interface UpdateCheckState {
  /** epoch ms of the last check we actually performed, or null if never */
  lastCheckedAt: number | null
  nowMs: number
  online: boolean
  /** document.visibilityState === 'visible' */
  visible: boolean
  /** why we are asking: a timer tick, or the page coming back to the foreground */
  reason: 'interval' | 'visible' | 'registered'
  intervalMs?: number
}

/**
 * Should we call registration.update() right now?
 *
 * `registered` always passes: that check is the one the browser would do anyway. usePwa does not
 * currently pass it — it stamps the baseline directly in onRegisteredSW, which is the same thing
 * without a call — so this branch exists for a caller that wants the check itself. Said plainly here
 * because the comment used to claim usePwa's baseline came through this branch, and it does not.
 */
export function shouldCheckForUpdate(s: UpdateCheckState): boolean {
  if (s.reason === 'registered') return true
  if (!s.online) return false
  if (!s.visible) return false
  if (s.lastCheckedAt === null) return true
  const intervalMs = s.intervalMs ?? SW_UPDATE_INTERVAL_MS
  const since = s.nowMs - s.lastCheckedAt
  // A clock that moved backwards (or a sleeping phone whose timers fire late) must
  // not be able to make this either permanently due or permanently blocked.
  if (since < 0) return true
  return since >= intervalMs
}

/**
 * How stale the running bundle is, in the words a person uses — for the update
 * prompt, so "a new version is available" can say how long they have been on the
 * old one. Null when we do not know, which must not be dressed up as "just now".
 */
export function bundleAgeText(loadedAtMs: number | null, nowMs: number): string | null {
  if (loadedAtMs === null || !Number.isFinite(loadedAtMs)) return null
  const s = (nowMs - loadedAtMs) / 1000
  if (s < 0) return null
  if (s < 90) return 'just now'
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  if (s < 172800) return `${(s / 3600).toFixed(1)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

export interface ReloadImpact {
  /** true when something is running: this reload is not free */
  busy: boolean
  /** the sentence under the reload button */
  text: string
}

/**
 * What reloading COSTS right now, in this fleet's current state.
 *
 * Auto-update is refused for a stated reason (see vite.config.ts and the note above): a reload
 * mid-task tears down the camera sockets and the run form of a robot that is moving. The manual
 * prompt inherited exactly that hazard and described it with one static sentence — "a running task
 * keeps running" — which it printed whether anything was running or not. So the toast said the same
 * thing at the safest moment and the worst one, and the operator, who is the person the decision was
 * deliberately left to, was given nothing to decide WITH.
 *
 * The app already knows: App tracks busyPeers for the wake lock. So name what is running, and when
 * nothing is, say that too — "a good moment" is the most useful thing this toast can tell someone who
 * has been putting off a reload beside a moving arm.
 *
 * What a reload does and does not do, stated once: the task itself runs in the ROBOT's process on the
 * mesh, so it survives. The camera websockets and anything typed into a form do not.
 */
export function reloadImpact(runningPeerIds: readonly string[]): ReloadImpact {
  const names = [...new Set(runningPeerIds.map(n => (n ?? '').trim()).filter(Boolean))]
  if (names.length === 0) {
    return {
      busy: false,
      text: 'Nothing is running right now — a good moment to reload. Camera streams reconnect by themselves.',
    }
  }
  const who = names.length === 1 ? names[0]
    : names.length === 2 ? `${names[0]} and ${names[1]}`
    : `${names[0]}, ${names[1]} and ${names.length - 2} more`
  return {
    busy: true,
    text: `${who} ${names.length === 1 ? 'is' : 'are'} running — the task itself keeps running on the robot, `
      + 'but reloading drops the camera streams and anything typed into a form. Between runs is safer.',
  }
}
