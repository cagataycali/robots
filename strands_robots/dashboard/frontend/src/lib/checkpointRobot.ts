/**
 * Which registry robot a checkpoint search should favour, read off the peer.
 *
 * A child sim peer is named `<parent>__<robot>`, so the robot is the suffix; a
 * Simulation peer lists its bodies in `sim_robots`; a hardware peer's `hw` line
 * starts with its registry name ("so101 @ /dev/tty..."). Anything else yields
 * '' and the search is unranked, which is honest: guessing a robot would put the
 * wrong checkpoints first.
 */
import type { Presence } from '../types'

export function robotHint(peerId: string, presence: Presence | null | undefined): string {
  const child = peerId.split('__')[1]
  if (child) return child
  const body = presence?.sim_robots?.[0]
  if (body) return body
  const hw = String(presence?.hw ?? '').trim().split(/[\s@]/)[0]
  if (hw && /^[a-z0-9_.-]+$/i.test(hw)) return hw
  return ''
}

/** Keyboard navigation over a listbox: the next active index for a key, or null when the key is not ours. */
export function nextActive(key: string, active: number, count: number): number | null {
  if (count <= 0) return null
  switch (key) {
    case 'ArrowDown': return active < 0 ? 0 : Math.min(active + 1, count - 1)
    case 'ArrowUp': return active <= 0 ? 0 : active - 1
    case 'Home': return 0
    case 'End': return count - 1
    default: return null
  }
}
