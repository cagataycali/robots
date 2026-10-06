/** What a record panel may honestly say when no camera tile can be drawn. */

export type CameraEvidence =
  | { kind: 'ok'; cams: string[] }
  | { kind: 'mute'; announced: string[]; message: string }
  | { kind: 'dropped'; requested: string[]; message: string }
  | { kind: 'unannounced'; message: string }

export interface CameraFailure { name: string; reason: string }

/**
 * The cameras this robot was configured with that did not open at connect, each with the
 * driver's own reason (`presence.camera_failures`). A camera whose frames are arriving is not
 * listed, whatever an older presence still says about it.
 */
export function cameraFailures(
  failures: Record<string, string> | undefined,
  arrived?: string[] | undefined,
): CameraFailure[] {
  const live = new Set((arrived ?? []).filter(Boolean))
  return Object.entries(failures ?? {})
    .filter(([name, reason]) => name && !live.has(name) && typeof reason === 'string')
    .map(([name, reason]) => ({ name, reason: reason.trim() || 'no reason given' }))
}

export function cameraEvidence(
  peerId: string,
  announced: string[] | undefined,
  arrived: string[] | undefined,
  requested?: string[] | undefined,
  failures?: Record<string, string> | undefined,
): CameraEvidence {
  const frames = (arrived ?? []).filter(Boolean)
  if (frames.length > 0) return { kind: 'ok', cams: frames }

  // The robot's own word on a camera that did not open beats every inference below: it names
  // the camera and what the device did, so "why is there no camera" has an answer, not a list
  // of three possibilities.
  const failed = cameraFailures(failures)
  if (failed.length > 0) {
    return {
      kind: 'dropped',
      requested: failed.map(f => f.name),
      message:
        failed.map(f => `${f.name}: ${f.reason}`).join('\n') +
        `\nRecording now would capture joints only. Reconfigure the camera from ${peerId}'s cameras sheet.`,
    }
  }

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
  // The dashboard's own spawn request (snapshot annotation `cameras_requested`, managed peers
  // only) is the ONLY thing that separates a joints-only robot from cameras that failed:
  // hardware_robot drops a camera it cannot open at connect, so presence is silent either way.
  const asked = (requested ?? []).filter(Boolean)
  if (asked.length > 0) {
    const list = asked.join(', ')
    return {
      kind: 'dropped',
      requested: asked,
      message:
        `${peerId} was started with ${list} but announces no cameras — they were dropped when it ` +
        `connected, which means the camera could not be opened: blocked by macOS privacy, held by ` +
        `another process, or unplugged. Recording now would capture joints only. ${peerId}'s log ` +
        `(devices › logs) names the one that failed.`,
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
 * The same verdict as a stage placeholder: a heading, one line under it, and the full sentence
 * for `title`.
 */
export function cameraPlaceholder(ev: CameraEvidence): { head: string; sub: string; title: string } | null {
  if (ev.kind === 'ok') return null
  if (ev.kind === 'dropped') {
    return {
      head: ev.requested.length === 1 ? `${ev.requested[0]} dropped` : 'cameras dropped',
      sub: `${ev.requested.join(', ')} requested, none opened`,
      title: ev.message,
    }
  }
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
