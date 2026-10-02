/** The extra a spawn refusal names, and the words the install button shows for it.
 *
 * The server answers a real-mode spawn whose child would die on an import with a 412
 * carrying `missing_extra` (the preflight), and a child that still died on one with a
 * 200 whose body carries `missing_extra` parsed from its refusal. Both shapes land
 * here; the Devices sheet turns them into one install button. The extra name is the
 * server's: it was graded against the package's declared extras there, and the button
 * sends it straight back, never a package name.
 */
export type MissingExtra = { extra: string; remedy: string | null; driver: string | null }

/** The install state the panel polls. Mirrors `GET /api/env/install/{id}`. */
export type InstallStatus = {
  id: string
  extra: string
  status: 'running' | 'done' | 'failed'
  exit_code: number | null
  lines: string[]
}

function record(x: unknown): Record<string, unknown> | null {
  return x && typeof x === 'object' && !Array.isArray(x) ? (x as Record<string, unknown>) : null
}

/** Read `missing_extra` off a spawn body — a 200 result, or an error body whose detail is nested. */
export function missingExtra(body: unknown): MissingExtra | null {
  const top = record(body)
  if (!top) return null
  // A refused request carries its dict under `error` (this app) or `detail` (FastAPI);
  // a 200-with-error spawn carries it at the top level.
  const candidates = [top, record(top.error), record(top.detail)]
  for (const c of candidates) {
    if (!c) continue
    const extra = c.missing_extra
    if (typeof extra === 'string' && extra.trim()) {
      return {
        extra: extra.trim(),
        remedy: typeof c.remedy === 'string' && c.remedy.trim() ? c.remedy.trim() : null,
        driver: typeof c.driver === 'string' && c.driver.trim() ? c.driver.trim() : null,
      }
    }
  }
  return null
}

/** The install button's label: the exact spec the server will install. */
export function installLabel(extra: string): string {
  return `install strands-robots[${extra}]`
}

/** One sentence for the install's state, for the status line under the button. */
export function installSentence(run: InstallStatus | null, busy: boolean): string {
  if (!run) return busy ? 'starting the install…' : ''
  if (run.status === 'running') return `installing [${run.extra}]… ${run.lines.length} lines so far`
  if (run.status === 'done') return `installed [${run.extra}] — spawn again`
  return `install of [${run.extra}] failed (exit ${run.exit_code ?? '?'}) — read the log below`
}

/** The last few log lines, newest last, for the tail under the status line. */
export function installTail(run: InstallStatus | null, n = 8): string[] {
  if (!run) return []
  return run.lines.slice(Math.max(0, run.lines.length - n))
}

/** The sentence the Environment tab shows for one extra's row. */
export function extraRowSentence(row: { name: string; installed: boolean; missing: string[] }): string {
  if (row.installed) return 'installed'
  const shown = row.missing.slice(0, 4).join(', ')
  const more = row.missing.length > 4 ? ` and ${row.missing.length - 4} more` : ''
  return `missing ${shown}${more}`
}
