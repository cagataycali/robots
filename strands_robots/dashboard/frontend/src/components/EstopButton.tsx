/**
 * The fleet stop lives in its own fixed layer, above every drawer, sheet, dock
 * and toast.
 *
 * It used to sit inside the header, which is `position: sticky; z-index: 50` -
 * under all four overlay layers (70 backdrop, 80 drawer, 90 sheet, 100 gate).
 * Measured with a real hit test: the button was VISIBLE but UNCLICKABLE on 4 of
 * 9 screens (devices / activity / settings / mobile devices), and the elements
 * winning the hit test were fully transparent, so nothing on screen told the
 * operator their click had gone somewhere else - on the devices drawer it
 * re-enumerated USB instead. A safety control that only looks available is
 * worse than none: it collects the panicked clicks that should go to the power
 * switch.
 *
 * The wrapper is pointer-events:none so this layer intercepts nothing but the
 * button itself.
 */
/**
 * `posture` (lib/linkHealth.estopPosture) marks the button when this page cannot
 * currently deliver the stop. It is NEVER disabled: the verdict is a snapshot,
 * the socket may be back by the time the finger lands, and a disabled brake is
 * a brake that refuses a click it might have delivered. The degraded style +
 * title say what will happen and name the arms' power switch instead.
 */
export default function EstopButton({
  onClick, posture,
}: { onClick: () => void; posture?: { degraded: boolean; title: string } }) {
  const degraded = !!posture?.degraded
  return (
    <div className="estop-layer">
      <button
        className={`estop${degraded ? ' unreachable' : ''}`}
        onClick={onClick}
        title={posture?.title ?? 'Stop every robot on the mesh - keyboard shortcut: .'}
        aria-label={degraded
          ? 'Emergency stop: the link is down, so this may not reach the robots'
          : 'Emergency stop: stop every robot on the mesh'}
        aria-keyshortcuts="."
      >
        {degraded ? '🛑 STOP ALL ⚠' : '🛑 STOP ALL'}
      </button>
    </div>
  )
}
