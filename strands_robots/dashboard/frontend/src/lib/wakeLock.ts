/**
 * Whether to take or drop the screen wake lock — as a pure decision.
 *
 * The lock exists so a phone propped next to the arms does not sleep while a robot is moving: a
 * sleeping screen drops the camera sockets and the operator loses sight of a moving arm.
 *
 * THE THING THAT MAKES THIS MORE THAN A ONE-LINER (Q89): the browser RELEASES a screen wake lock by
 * itself whenever the document becomes hidden, and it does not give it back on return. usePwa took
 * the lock when a task started and App re-asks only when `anyRunning` CHANGES — so the first time the
 * operator switched apps or their phone locked for a second, the lock was gone for the rest of the
 * task, silently, exactly while they were away from a moving arm. A lock that is only ever requested
 * once is not a lock; like the update prompt in the same file, the answer has to be re-offered.
 *
 * So the desired state is kept, and this function is asked again on every visibility change.
 */

export type WakeAction = 'request' | 'release' | 'none'

export interface WakeState {
  /** what the app last asked for: is any robot running */
  want: boolean
  /** do we currently hold a lock */
  held: boolean
  /** document.visibilityState === 'visible' */
  visible: boolean
  /** navigator.wakeLock exists (absent on iOS Safari before 16.4, and on Firefox) */
  supported: boolean
}

export function wakeLockAction(s: WakeState): WakeAction {
  // No API to call. The caller shows this to the operator instead of pretending the screen is held.
  if (!s.supported) return 'none'
  if (s.want) {
    if (s.held) return 'none'
    // A request while hidden is REFUSED by the browser (it throws NotAllowedError). Asking anyway
    // would burn the request and leave `held` false with nothing to show for it — waiting for the
    // page to come back is what actually gets the lock.
    return s.visible ? 'request' : 'none'
  }
  // Releasing does not need visibility, and a lock left held after the task ends drains the battery
  // of the phone the operator is watching the next task on.
  return s.held ? 'release' : 'none'
}

/**
 * What to tell the operator about the screen. `held` is the truth from the API, not our intent, so
 * "sleep is prevented" is never claimed on a platform that refused.
 */
export function wakeLockNote(s: WakeState): string | null {
  if (!s.want) return null
  if (!s.supported) return 'this browser cannot keep the screen awake — sleep may drop the camera view'
  return s.held ? null : 'screen sleep is not being prevented yet'
}
