import { cameraFailures } from '../lib/cameraEvidence'

/**
 * One row per configured camera that did not open at connect: its name, the driver's own
 * reason (what was asked, what the device answered), and - on a robot this dashboard spawned -
 * a button that opens the cameras sheet on that camera's row.
 */
export default function CameraFailures({ failures, arrived, onReconfigure }: {
  failures: Record<string, string> | undefined
  /** cameras whose frames are arriving; a live camera is never listed as dropped */
  arrived: string[]
  onReconfigure?: (cam: string) => void
}) {
  return (
    <>
      {cameraFailures(failures, arrived).map(f => (
        <div key={f.name} className="hint warn camfail" role="status">
          <span><b>{f.name}: dropped</b> — {f.reason}</span>
          {onReconfigure && (
            <button className="btn ghost" onClick={() => onReconfigure(f.name)}
                    title={`change ${f.name}'s mode and restart the robot`}>
              reconfigure
            </button>
          )}
        </div>
      ))}
    </>
  )
}
