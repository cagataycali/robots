import { useEffect, useState } from 'react'
import { api } from './endpoints'
import type { RegistryView } from '../types'

/** Polls `/api/mesh/iot/registry`. The server caches the AWS read for 30 s, so a 10 s poll here costs
 *  two API calls a minute at most. A failed poll keeps the last view: the registry panel must never
 *  take the rest of the fleet down with it. */
export function useRegistry(pollMs = 10_000, enabled = true): RegistryView | null {
  const [view, setView] = useState<RegistryView | null>(null)
  useEffect(() => {
    if (!enabled) return
    let alive = true
    const tick = async () => {
      try {
        const v = await api<RegistryView>('/api/mesh/iot/registry')
        if (alive) setView(v)
      } catch {
        /* keep the last view */
      }
    }
    void tick()
    const id = setInterval(tick, pollMs)
    return () => { alive = false; clearInterval(id) }
  }, [pollMs, enabled])
  return view
}

/** The Things that deserve a card of their own: provisioned, but no peer card of that name. A stale
 *  peer already has a greyed card with its own "last seen", so it is excluded too; two cards for one
 *  robot would read as two robots. */
export function registryCards(view: RegistryView | null, peerIds: Iterable<string>): RegistryView['things'] {
  if (!view || view.status !== 'ok') return []
  const known = new Set(peerIds)
  return view.things.filter(t => !t.peer_live && !t.heard_by_bridge && !known.has(t.thing_name))
}
