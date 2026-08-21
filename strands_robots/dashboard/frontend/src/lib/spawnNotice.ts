/**
 * The warning a SUCCESSFUL spawn can carry — currently the calibration gap.
 *
 * `/api/devices/spawn` answers 200 with a pid when the child starts, and the child starting is not
 * the same thing as the arm working: an arm spawned under a `robot_id` that has no calibration where
 * lerobot looks comes up publishing presence with ZERO joints, and until now the reason existed only
 * in that child's log. The server now says it in the spawn body (`calibration_warning`), so this is
 * the rule for turning that into something on screen without dressing a success as a failure.
 *
 * The rule is deliberately narrow: only a non-empty STRING the server actually sent becomes a
 * notice. A `true`, a number or an object would mean "something is wrong but I cannot tell you what",
 * and rendering that is worse than silence — it spends the operator's attention and returns nothing
 * they can act on.
 */
export type SpawnNotice = { text: string; tone: 'warn' }

export function spawnNotice(body: unknown): SpawnNotice | null {
  if (!body || typeof body !== 'object') return null
  const raw = (body as Record<string, unknown>).calibration_warning
  if (typeof raw !== 'string') return null
  const text = raw.trim()
  if (!text) return null
  return { text, tone: 'warn' }
}
