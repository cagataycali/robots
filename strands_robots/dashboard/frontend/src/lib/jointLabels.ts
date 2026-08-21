import { isOneSidedJoint, type JointUnit } from './jointScale'

/**
 * Reading the joint strip — UX_REVIEW #4, the part still missing.
 *
 * The numbers and the sparklines landed already. What a human still could not
 * answer from the pixels: **what unit is that, and what are those two graphics?**
 * `12.1` next to `-96.1` next to `0.40` is degrees, degrees and a gripper on its
 * own scale; the only explanation lived in `title=` attributes, which do not
 * exist on a touch screen and are not read out in a glance.
 *
 * Rules, both pinned by tests:
 *
 * - **The unit is reported, never fabricated per row.** A `servo` stream mixes
 *   degrees with a 0…100 gripper, so writing `°` on every value would be a lie
 *   on at least one row. The legend states the strip's unit verdict once, in
 *   words, including the mixed case.
 * - **A humanised joint name may never collide.** `shoulder_pan` → `Shoulder
 *   pan` is a readability win; two distinct keys collapsing to one label on a
 *   machine with two shoulders is a safety loss. `humanJointNames` hands back
 *   the RAW keys for any set that would collide, so the strip is either
 *   readable or literal — never ambiguous.
 */

/** `shoulder_pan.pos` → `Shoulder pan`. Never translates, only reformats. */
export function humanJointName(key: string): string {
  const bare = key.replace(/(_pos|\.pos)$/, '')
  const words = bare.replace(/[_.]+/g, ' ').replace(/\s+/g, ' ').trim()
  if (!words) return key
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/**
 * Labels for a whole strip. Returns raw keys when humanising would make two
 * rows read the same — an ambiguous label on a joint row is worse than an ugly
 * one, because the operator is matching a number to a limb.
 */
export function humanJointNames(keys: readonly string[]): string[] {
  const human = keys.map(humanJointName)
  const collides = new Set(human).size !== human.length
  return collides ? keys.slice() : human
}

/**
 * One line under the strip: what the numbers are in, and what the two graphics
 * mean. `jointNames` is used only to say whether a gripper is among them, since
 * that is the row whose unit differs.
 */
export function stripLegend(
  unit: JointUnit,
  windowMs: number,
  jointNames: readonly string[] = [],
): string {
  const secs = Math.round(windowMs / 1000)
  // ONE definition of "this joint is on its own 0…100 scale": jointScale's, the module that
  // actually puts it there. This line used to carry its own regex (/gripper|jaw|hand/i) and the two
  // disagreed in both directions — a joint named `claw` or `finger` got the one-sided scale with a
  // legend that never mentioned it, and a joint named `hand` got a legend promising a 0…100 scale
  // that no bar was drawn on. A legend that explains the scale rule must ASK the scale rule.
  const hasGripper = jointNames.some(isOneSidedJoint)
  const units = unit === 'radian'
    ? 'values in radians'
    : hasGripper
      ? 'values in degrees, gripper on its own 0…100 scale'
      : 'values in degrees'
  return `${units} · bar = position within travel seen so far · line = last ${secs}s`
}
