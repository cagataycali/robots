import { useEffect, useState } from 'react'

import { api } from '../lib/endpoints'
import { DISMISS_KEY, HintBody, lanHintVerdict, readDismissed } from '../lib/lanHint'

export default function LanHint() {
  const [body, setBody] = useState<HintBody | null>(null)
  const [dismissed, setDismissed] = useState<string[]>(() =>
    readDismissed(typeof localStorage === 'undefined' ? null : localStorage),
  )

  useEffect(() => {
    let alive = true
    // A network-topology hint is never worth an error toast: on any failure (404 from a
    // server that predates it, 401 before login, offline) it stays quiet.
    api<HintBody>('/api/network/hint')
      .then(b => { if (alive) setBody(b) })
      .catch(() => {})
    return () => { alive = false }
  }, [])

  const verdict = lanHintVerdict({
    body,
    origin: typeof location === 'undefined' ? '' : location.origin,
    dismissed,
  })
  if (!verdict.show) return null

  const dismiss = () => {
    const next = [...dismissed, verdict.url]
    setDismissed(next)
    try { localStorage.setItem(DISMISS_KEY, JSON.stringify(next)) } catch { /* private mode */ }
  }

  return (
    <div className="lan-hint" role="status" style={{ gridColumn: '1 / -1' }}>
      <span aria-hidden="true">&#127968;</span>
      <span className="lan-hint-text">{verdict.text}</span>
      {/* A plain link, not a redirect: leaving https for http is the operator's choice to
          make, and a silent downgrade would be indefensible. */}
      <a className="lan-hint-go" href={verdict.url}>open the local address</a>
      <button className="lan-hint-dismiss" onClick={dismiss} aria-label="dismiss this hint">
        &times;
      </button>
    </div>
  )
}
