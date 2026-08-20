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
