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
