/**
 * U16's last mile: the generated Python file reaches the operator's disk.
 *
 * Q123. POST /api/deploy/snippet has existed since 97b457cb and had ZERO callers — the PLAN row
 * ticked U16 as done citing 50d30c90, which is a 13-line handoff NOTE, so the richest generator in
 * the backend (it mirrors the live mesh posture, the camera config, and since Q122 refuses an
 * unreachable hub address) could not be reached from any screen. The directive was explicit: select
 * devices in the dashboard, get a snippet you can run on an edge device, with copy and download.
 *
 * These are the rules that are worth testing without a browser: what the file is called, and what a
 * refusal means to the person who clicked.
 */

/** A filename safe on every OS the operator might carry this to, derived from the server's own. */
export function safeFilename(name: string | null | undefined, peerId?: string | null): string {
  const raw = (name ?? '').trim() || `${(peerId ?? 'robot').trim() || 'robot'}.py`
  // Peer ids legitimately contain `__` (parent__child) and may contain a slash or a colon from a
  // hand-written config; a slash in a download name silently becomes a DIRECTORY the browser will
  // not create, so the download fails with no message at all.
  const base = raw.replace(/[/\\:*?"<>|\s]+/g, '-').replace(/^[.-]+/, '')
  const stem = base.endsWith('.py') ? base.slice(0, -3) : base
  return `${stem || 'robot'}.py`
}

/**
 * What the operator is told when the snippet cannot be produced. The server's 422 for a serial with
 * no remembered profile is the common case and it has a remedy, so it must not read as a bug.
 */
export function snippetRefusal(status: number, detail: string): string {
  const text = (detail || '').trim()
  if (status === 404 || /no profile remembered/i.test(text)) {
    return 'nothing is remembered for this board yet — spawn it once and the dashboard can write the file'
  }
  if (status === 422) return text || 'the saved payload is not enough to write a snippet'
  if (status === 401 || status === 403) return 'your session expired — reload and sign in again'
  return text || 'could not write the snippet'
}

/* ------------------------------------------------------------------------------------------------
 * Q122's other half: the operator can READ that the hub address was refused, and could not FIX it.
 *
 * The snippet runs on ANOTHER machine, so it needs the address that machine can use to reach this
 * dashboard's zenoh hub. The server guesses from the address the browser used and REFUSES a guess
 * that would mislead — reaching the dashboard through the Cloudflare tunnel yields a host with no
 * zenoh port, and a wrong address is worse than none because it looks authoritative and fails on a
 * machine the operator is not watching (Q122, cde0146a). So the snippet correctly ships with the
 * ZENOH_CONNECT line omitted and a `# NOTE:` explaining why.
 *
 * Then it stopped. /api/deploy/snippet has ALWAYS accepted an explicit `hub_host` in its body and
 * uses it verbatim (an operator's override deliberately outranks a guess), but the only caller sent
 * `{serial}` — so the one person who knows their LAN address had no field to type it in. Same defect
 * as Q54's fps and the record-refusal flags: the server accepts what the form cannot send.
 *
 * This layer deliberately does NOT re-judge the address. The server owns that judgement, and it
 * refuses public/loopback only when GUESSING; second-guessing an explicit override here would be a
 * second source of truth and would block the operator who knows better than both of us. What it
 * does is reject input that cannot be a host at all, with a reason.
 */

/** The server's own note when the snippet went out with NO hub address, else null. */
export function hubAddressMissing(code: string | null | undefined): string | null {
  if (!code) return null
  // Presence of the real line is the proof an address made it in - looking for the note alone would
  // keep prompting after a successful override, since the note stays for other reasons.
  if (/^\s*os\.environ\.setdefault\("ZENOH_CONNECT"/m.test(code)) return null
  const note = code.match(/^#\s*NOTE:\s*(.+?)\.?\s*$/m)
  return note
    ? note[1]
    : 'this file carries no hub address, so the peer it starts will connect to nothing'
}

export interface HubHostDraft {
  /** the address to send, or null when it cannot be one */
  host: string | null
  /** why it was rejected — '' when accepted */
  why: string
}

/** Tidy what the operator typed into something sendable, or say why it is not an address. */
export function cleanHubHost(input: string | null | undefined): HubHostDraft {
  const raw = (input ?? '').trim()
  if (!raw) return { host: null, why: 'type the address this dashboard is reachable at from the other machine' }
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) {
    return { host: null, why: 'paste a host, not a URL — no http:// and no path' }
  }
  const host = raw.replace(/\/+$/, '')
  if (/\s/.test(host)) return { host: null, why: 'an address has no spaces in it' }
  // host or host:port — a bare port, an empty port and a non-numeric port are all typos worth
  // catching here, because the snippet would otherwise be generated around them silently.
  // The port half is captured loosely and CHECKED below on purpose: matching only digits here
  // made 'gpu.lan:zenoh' fall through to "that is not a host name", which blames the wrong half
  // of what the operator typed.
  const m = host.match(/^([^:]+)(?::([^:]*))?$/)
  if (!m || !m[1]) return { host: null, why: 'that is not a host name or IP address' }
  if (m[2] !== undefined && !/^\d{1,5}$/.test(m[2])) {
    return { host: null, why: 'the part after ":" must be a port number' }
  }
  return { host, why: '' }
}
