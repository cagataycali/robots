/**
 * The exact `lerobot-calibrate` command for one connected arm.
 *
 * Calibration moves the arm through its range, so it stays a terminal job and
 * the dashboard commands no motion — that constraint is deliberate and is not
 * what this file changes. What it changes is the GUESSING: the old UI said
 * "run lerobot-calibrate in a terminal" and left the operator to invent four
 * flags, while the dashboard already knew every one of them (the port and
 * serial from the devices scan, the role from the U2 servo-bus measurement,
 * the model from the spawn registry).
 *
 * THE RULE OF THIS FILE, inherited from armPairing.ts: a confident wrong
 * command is worse than no command. Calibrating an arm as the wrong role
 * writes a calibration file under the wrong model/id — the operator then wires
 * a 12V follower against limits recorded for a 7.4V leader. So an unmeasured
 * port yields NO command line and a reason, never a plausible default.
 *
 * The role decides more than a word: on lerobot a follower is a `robots`
 * device and a leader is a `teleoperators` device (lerobot_calibrate.py:76),
 * so the wrong role also writes into the wrong directory tree.
 */

/** Roles the servo-bus measurement can report (arm_roles.py). */
export type ArmRole = 'follower' | 'leader' | 'unpowered' | 'mixed' | 'unknown'

export interface PortFacts {
  /** the OS device path, e.g. /dev/cu.usbmodem5AB01584281 */
  device: string
  serial_number?: string | null
  /** measured role; ABSENT means nobody measured it — not "unknown" */
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
}

/** What the UI should render for one port. */
export interface CalibratePlan {
  /** the command to copy, or null when we refuse to guess one */
  command: string | null
  /** always present: why this command, or why there is none */
  reason: string
  /** the flag values, so the UI can show them as fields if it wants */
  deviceType?: 'robots' | 'teleoperators'
  deviceModel?: string
  deviceId?: string
  /** true when the operator must measure the bus first (offer that button) */
  needsMeasurement?: boolean
}

/** lerobot's device_type is a function of the ROLE, not of the arm. */
function typeForRole(role: 'follower' | 'leader'): 'robots' | 'teleoperators' {
  return role === 'follower' ? 'robots' : 'teleoperators'
}

/**
 * `so101` + follower -> `so101_follower`. The family comes from the robot
 * registry (what the arm was spawned as), never from a hardcoded default: a
 * dashboard that assumes so101 will hand an so100 owner a command that writes
 * the wrong calibration file.
 */
export function deviceModel(family: string, role: 'follower' | 'leader'): string {
  const base = family.trim().toLowerCase()
  // Already role-qualified (the registry sometimes carries so101_follower).
  if (base.endsWith('_follower') || base.endsWith('_leader')) return base
  return `${base}_${role}`
}

/**
 * The id the calibration file is saved under. lerobot uses it as a FILE NAME,
 * so a serial is the stable choice (two so101 followers on one machine would
 * otherwise overwrite each other's calibration — the exact bug that makes an
 * arm move to another arm's limits).
 */
export function deviceId(facts: PortFacts, role: 'follower' | 'leader'): string {
  const serial = (facts.serial_number ?? '').trim()
  return serial ? `${role}_${serial}` : role
}

/** Shell-quote a value only when it needs it — a path with no spaces reads better bare. */
function shellArg(v: string): string {
  return /^[A-Za-z0-9_./:@=-]+$/.test(v) ? v : `'${v.replace(/'/g, `'\\''`)}'`
}

/**
 * Build the plan for one port.
 *
 * @param facts   the port row from GET /api/devices
 * @param family  the robot family this arm is (e.g. "so101") — from the registry
 */
export function calibratePlan(facts: PortFacts, family: string | null | undefined): CalibratePlan {
  const role = (facts.role ?? '').trim().toLowerCase()

  if (!role) {
    return {
      command: null,
      needsMeasurement: true,
      reason:
        'nobody has measured this bus yet, and the role decides both the model name and ' +
        'which directory the calibration is written to — measure the role first, then this ' +
        'command fills itself in',
    }
  }
  if (role === 'unpowered') {
    return {
      command: null,
      needsMeasurement: true,
      reason:
        `this bus reads ${facts.role_volts ?? '~5.5'}V, which is the USB logic rail — the arm's ` +
        'power supply is off. Calibration has to move the arm, and an unpowered arm cannot ' +
        'hold position, so switch its supply on and measure again',
    }
  }
  if (role === 'mixed') {
    return {
      command: null,
      reason:
        'the servos on this bus disagree about their voltage, which is a fault rather than a ' +
        'role — calibrating would record limits from a bus that is not answering consistently',
    }
  }
  if (role !== 'follower' && role !== 'leader') {
    return {
      command: null,
      needsMeasurement: true,
      reason: `the measurement came back "${role}", so the role is not established yet`,
    }
  }

  const fam = (family ?? '').trim()
  if (!fam) {
    return {
      command: null,
      reason:
        'the arm family is unknown here (so101, so100, …) and it is half of the model name — ' +
        'spawn the arm or pick its type first, so the command names the real model',
    }
  }

  const deviceType = typeForRole(role)
  const model = deviceModel(fam, role)
  const id = deviceId(facts, role)
  const command =
    `lerobot-calibrate --device_type=${deviceType} --device_model=${model} ` +
    `--device_id=${shellArg(id)} --port=${shellArg(facts.device)}`

  const measured =
    facts.role_volts != null
      ? `measured ${facts.role_volts}V on the servo bus`
      : `role recorded as ${role}`
  return {
    command,
    deviceType,
    deviceModel: model,
    deviceId: id,
    reason:
      `${measured}, so this arm is the ${role} — a ${role} is a lerobot "${deviceType}" device. ` +
      'Run this in a terminal: it will ask you to move the arm through its range by hand.',
  }
}
