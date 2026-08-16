import { useEffect, useRef, useState } from 'react'

interface CheckpointRow {
  repo_id: string
  local: boolean
  downloads: number | null
  policy_type: string | null
  tags: string[]
}

/**
 * Type-ahead over LeRobot policy checkpoints for `pretrained_name_or_path`.
 *
 * The registry names the FIELD but not its values - "lerobot_local" alone
 * says nothing about the thousands of public checkpoints (smolvla/act/pi0/…)
 * or the ones already sitting in the local HF cache. This widget searches
 * `/api/checkpoints/search` (local cache merged with a Hub search ranked by
 * downloads) as the user types, and selecting a row also reports the
 * checkpoint's `policy_type` so the caller can prefill lerobot_async's other
 * required field.
 */
export default function CheckpointPicker({ value, onPick, disabled }: {
  value: string
  onPick: (repoId: string, policyType: string | null) => void
  disabled?: boolean
}) {
  const [query, setQuery] = useState(value)
  const [rows, setRows] = useState<CheckpointRow[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const debounce = useRef<ReturnType<typeof setTimeout>>()
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => { setQuery(value) }, [value])

  // close on outside click
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const searchNow = (q: string) => {
    clearTimeout(debounce.current)
    debounce.current = setTimeout(async () => {
      setLoading(true)
      try {
        const r = await fetch(`/api/checkpoints/search?q=${encodeURIComponent(q)}&limit=12`)
        const j = await r.json()
        setRows(j.results ?? [])
        setOpen(true)
      } catch { setRows([]) }
      setLoading(false)
    }, 300)
  }

  const fmt = (n: number | null) =>
    n == null ? '' : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

  return (
    <div className="ckpt" ref={rootRef}>
      <input
        placeholder="search checkpoints… (e.g. smolvla, act so101)"
        value={query}
        onChange={e => { setQuery(e.target.value); onPick(e.target.value, null); searchNow(e.target.value) }}
        onFocus={() => { if (rows.length) setOpen(true); else searchNow(query) }}
        disabled={disabled}
      />
      {loading && <span className="ckpt-spin">…</span>}
      {open && rows.length > 0 && (
        <div className="ckpt-menu">
          {rows.map(r => (
            <button
              key={r.repo_id}
              className="ckpt-row"
              onMouseDown={e => e.preventDefault()}
              onClick={() => { onPick(r.repo_id, r.policy_type); setQuery(r.repo_id); setOpen(false) }}
            >
              <span className="ckpt-id">{r.repo_id}</span>
              <span className="ckpt-meta">
                {r.local && <b className="ckpt-local">local</b>}
                {r.policy_type && <em>{r.policy_type}</em>}
                {r.downloads != null && <span>↓{fmt(r.downloads)}</span>}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
