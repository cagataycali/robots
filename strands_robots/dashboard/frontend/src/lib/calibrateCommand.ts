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
  /**
   * The calibration id this port's SPAWN PROFILE already carries
   * (`/api/devices/profiles` -> `robot_id`). Absent when the port has never
   * been spawned. When present it is the id the running robot actually LOADS,
   * so it outranks anything this module could invent - see `deviceId`.
   */
  robot_id?: string | null
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
  /**
   * Present when the id in the command deserves a sentence of its own: it came
   * from the arm's profile, or it is new, or its NAME contradicts the measured
   * role. Never a refusal - the id is still correct.
   */
  idNote?: string
  /**
   * True only for the case that can mislead a human: the id is correct but its
   * NAME says the other role. A boolean, so the UI never has to pattern-match
   * on prose to decide whether to warn.
   */
  idWarn?: boolean
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
 * The id the calibration file is saved under. lerobot uses it as a FILE NAME.
 *
 * A PROFILE'S `robot_id` WINS whenever the port has one, and that is the whole
 * point: the spawned robot loads its calibration by that id, so a command that
 * invents a different one sends the operator through the full ceremony - moving
 * every joint to its limits by hand - and writes a file the arm will never
 * read. Nothing would report a failure: the calibration succeeds, the arm keeps
 * running on its old limits, and the only symptom is an arm still reaching
 * where it should not. Measured on this machine: so101-arm-2's profile carries
 * `leader_arm`, so the invented `follower_5AB0158428` would have been exactly
 * that dead end.
 *
 * With no profile there is nothing to honour, and then a serial-qualified name
 * is the safe invention: two so101 followers on one machine would otherwise
 * both be `follower` and overwrite each other's calibration.
 */
export function deviceId(facts: PortFacts, role: 'follower' | 'leader'): string {
  const known = (facts.robot_id ?? '').trim()
  if (known) return known
  const serial = (facts.serial_number ?? '').trim()
  return serial ? `${role}_${serial}` : role
}

/**
 * Why the command carries THIS id — and the trap worth naming out loud.
 *
 * An id is a file name, not a claim about the hardware, so an id named
 * `leader_arm` on a bus measured at 12.6V (a follower) is CORRECT to pass and
 * still worth a sentence: an operator who reads the name instead of the flags
 * concludes they are calibrating the other arm. That mislabel is the same one
 * cagatay originally reported on the record screen, one surface over.
 */
/* Module-private on purpose: both callers are in this file and both are tested through their own verdicts (idWarn); exporting the predicate alone invited a caller that skips the sentence explaining it. */
function idNameContradictsRole(id: string, role: 'follower' | 'leader'): boolean {
  const other = role === 'follower' ? 'leader' : 'follower'
  return id.trim().toLowerCase().includes(other)
}

export function idNote(facts: PortFacts, role: 'follower' | 'leader'): string | undefined {
  const known = (facts.robot_id ?? '').trim();
  if (!known) {
    const serial = (facts.serial_number ?? '').trim();
    return serial
      ? `this port has no spawn profile yet, so the id is built from its serial (${serial}) — ` +
        'spawn the arm with this same id afterwards, or lerobot will not find the calibration'
      : 'this port has no spawn profile and reports no serial, so the id is just the role — ' +
        'two arms of the same role on one machine would overwrite each other';
  }
  const contradicts = idNameContradictsRole(known, role);
  const base = `this is the id the arm already runs with (${known}), so the calibration lands where it will be read`;
  if (!contradicts) return base;
  const volts = facts.role_volts != null ? `${facts.role_volts}V` : 'its measured voltage';
  return (
    `${base} — note the id is NAMED "${known}" while this bus measures ${volts} = ${role}. ` +
    'The id is only a file name and is still the right one to pass; the name is what is wrong, ' +
    'so do not let it convince you this is the other arm'
  );
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
    idNote: idNote(facts, role),
    idWarn: idNameContradictsRole(id, role),
    reason:
      `${measured}, so this arm is the ${role} — a ${role} is a lerobot "${deviceType}" device. ` +
      'Run this in a terminal: it will ask you to move the arm through its range by hand.',
  }
}

/** One entry of `GET /api/devices/profiles` — only the fields this file needs. */
export interface SpawnProfile {
  robot_id?: string | null
  port?: string | null
  serial_number?: string | null
}

/**
 * The calibration id remembered for this physical port, or undefined.
 *
 * The profile store is keyed by SERIAL, which is the identity that survives
 * re-plugging (a `/dev/cu.usbmodem…` path can change, and on this machine one
 * legacy entry is keyed by the port string itself). So: serial first, then a
 * port match, and nothing at all rather than a guess — `deviceId` invents a
 * safe id when there is nothing to honour, and inventing is better than
 * honouring the WRONG arm's id, which would write one arm's limits under the
 * other's name.
 */
export function knownCalibrationId(
  profiles: Record<string, SpawnProfile> | null | undefined,
  facts: { device: string; serial_number?: string | null },
): string | undefined {
  if (!profiles) return undefined
  const serial = (facts.serial_number ?? '').trim()
  const bySerial = serial ? profiles[serial] : undefined
  const id = (bySerial?.robot_id ?? '').trim()
  if (id) return id
  // No serial-keyed entry: accept a port match, but only an EXACT one. A prefix
  // match would pair /dev/cu.usbmodem5AB01818061 with a different arm's port.
  for (const entry of Object.values(profiles)) {
    if ((entry?.port ?? '').trim() === facts.device) {
      const pid = (entry?.robot_id ?? '').trim()
      if (pid) return pid
    }
  }
  return undefined
}
