import { useEffect, useState } from 'react'
import { api } from '../lib/endpoints'
import { extraRowSentence } from '../lib/missingExtra'
import InstallExtra from './InstallExtra'

type ExtraRow = { name: string; installed: boolean; missing: string[]; count: number }
type EnvDoc = {
  python: string
  version: string
  installer: string
  editable_source: string | null
  extras: ExtraRow[]
}

/** The package's declared extras in the dashboard's interpreter, each with an install button.
 *
 * Read from `GET /api/env`; the names come from the server's own `Provides-Extra`, so the
 * button can only ever ask for one of them.
 */
export default function ExtrasList() {
  const [doc, setDoc] = useState<EnvDoc | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [installing, setInstalling] = useState<string | null>(null)

  const load = () =>
    api<EnvDoc>('/api/env')
      .then(d => { setDoc(d); setError(null) })
      .catch(e => setError(e?.message ?? String(e)))

  useEffect(() => { void load() }, [])

  return (
    <div className="extras">
      <h4>Python extras</h4>
      {doc && (
        <p className="hint">
          <code>{doc.python}</code> (Python {doc.version}, installs with <code>{doc.installer}</code>
          {doc.editable_source ? <>, editable from <code>{doc.editable_source}</code></> : null}).
          A spawn that needs an extra this interpreter lacks is refused with its name; install it here.
        </p>
      )}
      {error && <div className="result bad">⚠ {error}</div>}
      {doc && (
        <div className="extras-list">
          {doc.extras.map(row => (
            <div className="extra-row" key={row.name}>
              <code>[{row.name}]</code>
              <span className={row.installed ? 'hint' : 'hint warn'}>{extraRowSentence(row)}</span>
              {!row.installed && installing !== row.name && (
                <button className="btn ghost tiny" onClick={() => setInstalling(row.name)}>install</button>
              )}
              {installing === row.name && (
                <InstallExtra extra={row.name} onDone={() => { setInstalling(null); void load() }} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
