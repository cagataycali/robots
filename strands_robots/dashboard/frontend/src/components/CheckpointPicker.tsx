import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { api } from '../lib/endpoints'
import { emptyNote, isCurrent } from '../lib/checkpointSearch'
import { nextActive } from '../lib/checkpointRobot'
import type { PolicyFit } from './RunForm'

interface CheckpointRow {
  repo_id: string
  local: boolean
  downloads: number | null
  policy_type: string | null
  tags: string[]
  /** the search named the selected robot and this row carries its name */
  robot_match?: boolean
}

interface HfAuth { authenticated: boolean; user: string | null; detail: string | null }

/** What the fit pill says about one row: compared and fine, compared and refused, or not comparable. */
export type FitVerdict = 'fits' | 'mismatch' | 'unknown'

export function fitVerdict(fit: PolicyFit | null | undefined): FitVerdict {
  if (!fit || fit.evidence === false || !fit.checked?.length) return 'unknown'
  return fit.blocking ? 'mismatch' : 'fits'
}

/** How many rows the picker asks the fit route about per search: the ones on screen without a scroll. */
const FIT_ROWS = 6

/**
 * Type-ahead over LeRobot policy checkpoints for `pretrained_name_or_path`.
 *
 * A combobox: the input owns focus, ArrowUp/Down/Home/End move the active row,
 * Enter picks it, Escape closes. Rows that name the selected robot come first
 * (the server ranks them when `robot` is given), and the first rows carry the
 * policy-fit verdict against the selected peer so a mismatch is visible before
 * the pick, not after the run is refused.
 */
export default function CheckpointPicker({ value, onPick, disabled, robot, peerId }: {
  value: string
  onPick: (repoId: string, policyType: string | null) => void
  disabled?: boolean
  /** registry name of the robot being driven (ranks the results); '' = unranked */
  robot?: string
  /** the peer the fit is judged against; absent = no fit pills */
  peerId?: string
}) {
  const [query, setQuery] = useState(value)
  const [rows, setRows] = useState<CheckpointRow[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [hubProblem, setHubProblem] = useState<string | null>(null)
  const [hfAuth, setHfAuth] = useState<HfAuth | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [active, setActive] = useState(-1)
  const [fits, setFits] = useState<Record<string, FitVerdict>>({})
  const debounce = useRef<ReturnType<typeof setTimeout>>()
  // The debounce cancels a pending TIMER, not an in-flight fetch: without a sequence, a slow
  // search for "act" can resolve after a fast one for "smolvla" and paint act's rows under the
  // newer query.
  const seq = useRef(0)
  // Which query the rows on screen belong to, so the empty note cannot describe
  // a different search than the one that produced it.
  const [shownQuery, setShownQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const listId = useRef(`ckpt-list-${Math.random().toString(36).slice(2, 8)}`).current

  useEffect(() => { setQuery(value) }, [value])

  // close on outside click
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  // Fit pills for the first rows: one GET per (peer, repo), cached for the picker's life.
  useEffect(() => {
    if (!peerId || !open) return
    const want = rows.slice(0, FIT_ROWS).map(r => r.repo_id).filter(id => fits[id] === undefined)
    if (!want.length) return
    let alive = true
    for (const id of want) {
      void api(`/api/robots/${encodeURIComponent(peerId)}/policy-fit?repo_id=${encodeURIComponent(id)}`)
        .then((v: PolicyFit) => { if (alive) setFits(f => ({ ...f, [id]: fitVerdict(v) })) })
        // A failed lookup is not evidence of a mismatch: the pill stays "unknown".
        .catch(() => { if (alive) setFits(f => ({ ...f, [id]: 'unknown' })) })
    }
    return () => { alive = false }
  }, [rows, peerId, open, fits])

  const searchNow = (q: string) => {
    clearTimeout(debounce.current)
    const mine = ++seq.current
    debounce.current = setTimeout(async () => {
      setLoading(true)
      try {
        const robotParam = robot ? `&robot=${encodeURIComponent(robot)}` : ''
        const j = await api(`/api/checkpoints/search?q=${encodeURIComponent(q)}&limit=12${robotParam}`)
        if (!isCurrent(mine, seq.current)) return
        setRows(j.results ?? [])
        setHubProblem(j.hub_problem ?? null)
        setHfAuth(j.hf_auth ?? null)
        setFailed(null)
        setShownQuery(q)
        setActive(-1)
        setOpen(true)
      } catch (e) {
        // the search endpoint itself failed (auth, network) - name it instead
        // of rendering the same silence as 'no matches'
        if (!isCurrent(mine, seq.current)) return
        setRows([])
        setFailed((e as any)?.message ?? String(e))
        setShownQuery(q)
        setOpen(true)
      } finally {
        // A superseded request must not switch the spinner off under a newer one.
        if (isCurrent(mine, seq.current)) setLoading(false)
      }
    }, 300)
  }

  const pick = (r: CheckpointRow) => { onPick(r.repo_id, r.policy_type); setQuery(r.repo_id); setOpen(false); setActive(-1) }

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') { if (open) { e.preventDefault(); setOpen(false) } return }
    if (e.key === 'Enter') {
      if (open && active >= 0 && rows[active]) { e.preventDefault(); pick(rows[active]) }
      return
    }
    const next = nextActive(e.key, active, rows.length)
    if (next === null) return
    e.preventDefault()
    if (!open) { if (rows.length) setOpen(true); else searchNow(query) }
    setActive(next)
  }

  const fmt = (n: number | null) =>
    n == null ? '' : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

  const activeId = open && active >= 0 && rows[active] ? `${listId}-${active}` : undefined

  return (
    <div className="ckpt" ref={rootRef}>
      <input
        placeholder={robot ? `search checkpoints for ${robot}…` : 'search checkpoints… (e.g. smolvla, act so101)'}
        aria-label="search checkpoints"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={activeId}
        value={query}
        onChange={e => { setQuery(e.target.value); onPick(e.target.value, null); searchNow(e.target.value) }}
        onFocus={() => { if (rows.length) setOpen(true); else searchNow(query) }}
        onKeyDown={onKeyDown}
        disabled={disabled}
      />
      {loading && <span className="ckpt-spin">…</span>}
      {open && (
        <div className="ckpt-menu" id={listId} role="listbox" aria-label="checkpoints">
          {failed && <div className="ckpt-note bad">✗ search failed: {failed}</div>}
          {/* When there are no rows the empty note carries this reason itself — two lines saying "the Hub is down" is one line the eye skips. */}
          {!failed && hubProblem && rows.length > 0 && <div className="ckpt-note warn">⚠ {hubProblem}</div>}
          {!failed && hfAuth && (
            <div className={`ckpt-note ${hfAuth.authenticated ? 'ok' : ''}`}>
              {hfAuth.authenticated
                ? `HF: signed in as ${hfAuth.user} — private + gated repos reachable`
                : `HF: anonymous — ${hfAuth.detail ?? 'public repos only'}`}
            </div>
          )}
          {rows.length === 0 && !failed && (
            // Scoped to what was actually consulted: with the Hub down, only the
            // local cache answered, and "no checkpoints match" would be a claim
            // about a catalogue nobody asked.
            <div className={hubProblem ? 'ckpt-note warn' : 'ckpt-note'}>
              {emptyNote({ query: shownQuery, hubProblem })}
            </div>
          )}
          {rows.map((r, i) => {
            const fit = peerId ? fits[r.repo_id] : undefined
            return (
              <button
                key={r.repo_id}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={i === active}
                className={`ckpt-row${i === active ? ' active' : ''}`}
                onMouseDown={e => e.preventDefault()}
                onMouseEnter={() => setActive(i)}
                onClick={() => pick(r)}
              >
                <span className="ckpt-id">{r.repo_id}</span>
                <span className="ckpt-meta">
                  {r.local && <b className="ckpt-local">local</b>}
                  {r.robot_match && robot && <b className="ckpt-robot" title={`named after ${robot}`}>{robot}</b>}
                  {r.policy_type && <em>{r.policy_type}</em>}
                  {r.downloads != null && <span>↓{fmt(r.downloads)}</span>}
                  {fit && fit !== 'unknown' && (
                    <span className={`ckpt-fit ${fit}`} title={fit === 'fits' ? 'declared features match this robot' : 'declared features do not fit this robot'}>
                      {fit === 'fits' ? '✓ fits' : '✗ mismatch'}
                    </span>
                  )}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
