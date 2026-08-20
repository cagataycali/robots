/**
 * What a record panel may honestly say when no camera tile can be drawn.
 *
 * The fleet snapshot holds TWO different camera facts and the record view was
 * reading only one of them: `presence.cameras` is what the robot ANNOUNCES it
 * has, while `peer.cameras` is keyed by the frames that have actually ARRIVED.
 * The old copy branched on the arrivals alone and said "no cameras announced by
 * this peer - the dataset will have joints only", which is a flat contradiction
 * of the presence the same snapshot carries: on this Mac both arms announced
 * `top` and `wrist` while macOS blocked every frame (BUGS.md Q25), so the
 * sentence blamed the robot for a permission the operator could fix, and
 * promised a joints-only dataset as though it were a choice.
 *
 * Three worlds, and the difference matters before a recording, not after:
 *  - `ok`         - frames are here, draw the tiles.
 *  - `mute`       - announced, nothing arriving: a real failure to chase.
 *  - `unannounced`- the robot itself lists none. Still NOT proof it was started
 *                   without any: hardware_robot DROPS a camera it cannot open at
 *                   connect, so a blocked camera and an intentional joints-only
 *                   robot look identical from here. Say that rather than pick.
 *
 * Both warnings speak in the conditional ("would record"), because a recording
 * has not started yet and a camera that returns before ▶ makes any promise about
 * the dataset's contents false.
 */

export type CameraEvidence =
  | { kind: 'ok'; cams: string[] }
  | { kind: 'mute'; announced: string[]; message: string }
  | { kind: 'unannounced'; message: string }

export function cameraEvidence(
  peerId: string,
  announced: string[] | undefined,
  arrived: string[] | undefined,
): CameraEvidence {
  const frames = (arrived ?? []).filter(Boolean)
  if (frames.length > 0) return { kind: 'ok', cams: frames }

  const names = (announced ?? []).filter(Boolean)
  if (names.length > 0) {
    const list = names.join(', ')
    return {
      kind: 'mute',
      announced: names,
      message:
        `${peerId} announces ${names.length} camera${names.length > 1 ? 's' : ''} (${list}) ` +
        `but no frames have arrived — recording now would capture joints only. A camera that is ` +
        `blocked by macOS, held by another process or unplugged looks identical from here: open ` +
        `devices › logs for ${peerId} to see which.`,
    }
  }
  return {
    kind: 'unannounced',
    message:
      `${peerId} lists no cameras — recording now would capture joints only. That is either a ` +
      `deliberately joints-only robot or cameras that failed to open and were dropped when it ` +
      `connected; from here the two are indistinguishable, and ${peerId}'s log says which.`,
  }
}


/**
 * The same verdict as a stage placeholder: a heading, one line under it, and the
 * full sentence for `title`.
 *
 * RobotDetail's empty stage read `no camera / this peer publishes none` — the
 * detail screen is exactly where an operator goes to find out WHY a camera is
 * missing, so it was the worst place to state the one thing the snapshot cannot
 * know. `mute` gets a heading that describes the evidence (no FRAMES, which is
 * what is missing) instead of the hardware (no CAMERA, which was announced).
 *
 * Kept short because it renders inside a tile: the head is two words, the sub
 * line names the announced cameras or the ambiguity, and the untruncated
 * message goes to `title` so nothing is lost.
 */
export function cameraPlaceholder(ev: CameraEvidence): { head: string; sub: string; title: string } | null {
  if (ev.kind === 'ok') return null
  if (ev.kind === 'mute') {
    return {
      head: 'no frames',
      sub: `${ev.announced.join(', ')} announced, nothing arriving`,
      title: ev.message,
    }
  }
  return {
    head: 'no camera',
    sub: 'none listed — joints-only, or dropped at connect',
    title: ev.message,
  }
}
