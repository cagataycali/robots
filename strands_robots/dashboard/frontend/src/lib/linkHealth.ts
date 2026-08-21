/**
 * Is this dashboard still ATTACHED to the fleet — and if not, say so before the
 * operator finds out by clicking the brake. (JOURNEYS #16.)
 *
 * The measured hole: `:8090` went down twice in ~30 minutes of ordinary
 * development. When the socket drops while robots are already on screen, the
 * cards keep rendering the last snapshot, the joint strips hold their final
 * numbers, and 🛑 STOP ALL looks exactly like a working brake. The only signal
 * was the word `CLOSED` in the header. The device-has-no-network case already
 * got a toast that admits "commands will fail"; the far more common
 * server-is-gone case got nothing.
 *
 * Why that ordering is wrong: a frozen view is not merely stale, it is
 * MISLEADING IN A SPECIFIC DIRECTION. The last frame the operator saw is a
 * robot that was fine. The arms keep doing whatever they were told to do, and
 * the one control that would stop them is the one that cannot leave the
 * browser.
 *
 * This module is pure so the wording can be tested: what the operator reads in
 * that moment IS the feature.
 */

export type ConnState = 'connecting' | 'open' | 'closed' | 'unauthorized'

export interface LinkInput {
  conn: ConnState
  /** navigator.onLine — this DEVICE's network, a different failure from the server's */
  browserOnline: boolean
  /** epoch ms of the last frame received on the mesh socket, if any */
  lastEventAt?: number
  now: number
  /** how many robots are on screen right now (a frozen empty fleet misleads nobody) */
  peerCount: number
  /** true once the socket has been open at least once this session */
  everOpen: boolean
  /** silence after which an open-but-mute socket is suspicious (ms) */
  stallMs?: number
  /**
   * Q88: the rejection is OUR OWN sign-in having lapsed, not the server changing its mind.
   * Same facts, completely different sentence — "the server rejected this session" sends the
   * operator to the backend, and the measured incident was 19.3 hours of hunting a camera bug
   * for an expired token.
   */
  sessionExpired?: boolean
  /**
   * Q100: is the DASHBOARD's own mesh session up? A different failure from every other one here —
   * the API is fine, this page is connected to it, and nothing can reach a robot. The socket keeps
   * whatever frames it has, so the cards stay on screen and the only honest verdict comes from the
   * server saying so (`mesh.online`, already on the snapshot this page reads).
   *
   * `undefined` means the server did not say, which must change NOTHING: absence of evidence is not
   * evidence of a dead mesh, and inventing one would put a false brake warning on a working fleet.
   */
  meshOnline?: boolean
}

export interface LinkVerdict {
  kind: 'live' | 'connecting' | 'device-offline' | 'unauthorized' | 'lost' | 'stalled' | 'mesh-down'
  /** commands (including the e-stop) can be expected to reach the fleet */
  commandsWork: boolean
  /** true when robots are on screen that this dashboard can no longer command */
  misleading: boolean
  headline: string
  /** what it costs and what to do instead — the sentence that earns its place */
  detail: string
  /** for the e-stop's own title: why the brake will not work */
  estopReason?: string
}

const LIVE: LinkVerdict = {
  kind: 'live', commandsWork: true, misleading: false,
  headline: '', detail: '',
}

/** The physical fallback, named every time the brake cannot reach the arms. */
const PHYSICAL =
  'Use the arms\u2019 power switch — that is the only brake that does not go through this page.'

export function linkHealth(i: LinkInput): LinkVerdict {
  const stallMs = i.stallMs ?? 20000
  const showing = i.peerCount > 0

  // This device's network first: nothing at all can leave the machine, so no
  // amount of server health matters.
  if (!i.browserOnline) {
    return {
      kind: 'device-offline', commandsWork: false, misleading: showing,
      headline: 'This device is offline',
      detail: showing
        ? `The fleet below is a cached snapshot. Commands and 🛑 STOP ALL cannot leave this device. ${PHYSICAL}`
        : 'Commands cannot leave this device until its network is back.',
      estopReason: 'this device has no network — STOP ALL cannot be sent',
    }
  }

  if (i.conn === 'unauthorized') {
    if (i.sessionExpired) {
      return {
        kind: 'unauthorized', commandsWork: false, misleading: showing,
        headline: 'Your sign-in has expired',
        // The fix is one tap and it is on this page, so say that before anything else. The
        // e-stop caveat still has to be here: a brake that cannot leave the browser must never
        // be implied to work, however ordinary the cause.
        detail: `Sign in again to command the fleet — nothing is wrong with the robots, this page is `
          + `being refused. Until then 🛑 STOP ALL cannot be sent. ${PHYSICAL}`,
        estopReason: 'this sign-in has expired — STOP ALL will be refused until you sign in again',
      }
    }
    return {
      kind: 'unauthorized', commandsWork: false, misleading: showing,
      headline: 'The server rejected this session',
      detail: 'Every command, including 🛑 STOP ALL, will be refused until the token is accepted again.',
      estopReason: 'the server is rejecting this session — STOP ALL will be refused',
    }
  }

  // The API is up and this page is talking to it, and STILL nothing can reach a robot: the
  // dashboard's own mesh session is down. Only trusted while the socket is OPEN, because that is
  // when the flag is fresh news; after a drop, `conn` already tells the better story and a
  // remembered `false` would explain a live fleet with a dead one.
  //
  // THE MEASURED HOLE (Q100): App renders "the dashboard's mesh session is down" inside its
  // EMPTY-FLEET block, so it appears only when no robot is on screen. When the session dies with
  // cards already rendered — the case that misleads — that explanation never rendered, this module
  // was never told, and the two things it says at that moment were both wrong: "Commands should
  // still get through", and an e-stop button whose title read like a working brake.
  if (i.conn === 'open' && i.meshOnline === false) {
    return {
      kind: 'mesh-down', commandsWork: false, misleading: showing,
      headline: 'This dashboard is not on the robot mesh',
      detail: (showing
        ? 'The API is up and this page is connected to it, but its mesh session is down — the fleet '
          + 'below is the last thing the mesh reported and the robots keep doing whatever they were '
          + 'last told. No command, including 🛑 STOP ALL, can reach them. '
        : 'The API is up and this page is connected to it, but its mesh session is down, so no robot '
          + 'can be seen or commanded. ') + PHYSICAL,
      estopReason: "the dashboard's mesh session is down — STOP ALL cannot reach any robot",
    }
  }

  if (i.conn === 'open') {
    // An open socket that has gone quiet. Only suspicious when robots are on
    // screen: with an empty fleet there is legitimately nothing to send, and
    // crying "stalled" at an idle dashboard would train the operator to ignore
    // this banner — which is the one banner they must not learn to ignore.
    if (showing && i.lastEventAt !== undefined && i.now - i.lastEventAt > stallMs) {
      return {
        kind: 'stalled', commandsWork: true, misleading: true,
        headline: 'No fleet updates for a while',
        detail: `The socket is open but has sent nothing for ${Math.round((i.now - i.lastEventAt) / 1000)}s, `
          + 'so what you see below may no longer be true. Commands should still get through.',
      }
    }
    return LIVE
  }

  // 'connecting' before the first open is ordinary startup, not a warning.
  if (i.conn === 'connecting' && !i.everOpen) {
    return { ...LIVE, kind: 'connecting', commandsWork: false }
  }

  const frozenFor = i.lastEventAt !== undefined ? Math.round((i.now - i.lastEventAt) / 1000) : undefined
  const reconnecting = i.conn === 'connecting'
  return {
    kind: 'lost', commandsWork: false, misleading: showing,
    headline: reconnecting ? 'Reconnecting to the dashboard API…' : 'Disconnected from the dashboard API',
    detail: showing
      ? `The fleet below is frozen${frozenFor !== undefined ? ` (${frozenFor}s old)` : ''} and the robots keep doing `
        + `whatever they were last told. 🛑 STOP ALL cannot reach them from here. ${PHYSICAL}`
      : 'The dashboard API is unreachable from this browser.',
    estopReason: 'the dashboard API is unreachable — STOP ALL cannot be delivered',
  }
}

/**
 * Never DISABLE the stop, however dead the link looks.
 *
 * Same reasoning runRisk.ts records for a robot that may reconnect between
 * judgment and click: the verdict is a snapshot, the socket may be back by the
 * time the finger lands, and a disabled brake is a brake that refuses a click
 * it might have delivered. Warn, keep it live, and name the physical fallback.
 */
export function estopPosture(v: LinkVerdict): { degraded: boolean; title: string } {
  return v.commandsWork
    ? { degraded: false, title: 'Stop every robot on the mesh - keyboard shortcut: .' }
    : { degraded: true, title: `⚠ ${v.estopReason ?? 'the link is down'} — still worth pressing, it is sent the moment the link returns. ${PHYSICAL}` }
}
