import { useEffect, useRef, useState } from 'react'
import { api, post, HttpError } from '../lib/endpoints'
import { installLabel, installSentence, installTail, type InstallStatus } from '../lib/missingExtra'

type Props = {
  /** The declared extra to install; the server graded the name, this only sends it back. */
  extra: string
  /** Why it is offered — the preflight's remedy line or the child's refusal. */
  reason?: string | null
  /** Called once the install finished successfully, so the caller can re-enable spawn. */
  onDone?: () => void
}

/** One install of a declared extra into the dashboard's interpreter, with its log tail.
 *
 * Starts `POST /api/env/install {extra}`, then polls `GET /api/env/install/{id}` once a
 * second until the subprocess exits. The server runs one install at a time (409 while one
 * runs), so a second button press while another install is live reads that refusal.
 */
export default function InstallExtra({ extra, reason, onDone }: Props) {
  const [run, setRun] = useState<InstallStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current) }, [])

  const poll = (id: string) => {
    timer.current = window.setTimeout(async () => {
      try {
        const r = await api<InstallStatus>(`/api/env/install/${id}`)
        setRun(r)
        if (r.status === 'running') poll(id)
        else {
          setBusy(false)
          if (r.status === 'done') onDone?.()
        }
      } catch (e: any) {
        setBusy(false)
        setError(e?.message ?? String(e))
      }
    }, 1000)
  }

  const start = async () => {
    setBusy(true); setError(null); setRun(null)
    try {
      const r = await post<InstallStatus>('/api/env/install', { extra })
      setRun(r)
      poll(r.id)
    } catch (e: any) {
      setBusy(false)
      setError(e instanceof HttpError && e.status === 409
        ? `another install is still running — wait for it: ${e.message}`
        : (e?.message ?? String(e)))
    }
  }

  const tail = installTail(run)
  return (
    <div className="install-extra" role="group" aria-label={`install ${extra}`}>
      {reason && <div className="hint">{reason}</div>}
      <div className="row">
        <button className="btn go" disabled={busy} onClick={() => void start()}>
          {busy ? 'installing…' : installLabel(extra)}
        </button>
        <span className="hint" role="status">{installSentence(run, busy)}</span>
      </div>
      {error && <div className="result bad" role="alert">⚠ {error}</div>}
      {tail.length > 0 && (
        <pre className="install-log" aria-label="install log tail">{tail.join('\n')}</pre>
      )}
    </div>
  )
}
