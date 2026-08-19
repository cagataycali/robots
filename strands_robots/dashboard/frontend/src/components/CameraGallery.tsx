import { useEffect, useRef, useState } from 'react'
import { apiBlob } from '../lib/endpoints'

export interface CameraInfo {
  index: number
  width?: number
  height?: number
  fps?: number | null
  claimed_by?: string
}

export interface CameraName { listing_index: number; name: string }

/**
 * The devices screen's answer to "which camera is index N?".
 *
 * An OpenCV index is a position, never an identity: the OS lists camera
 * *names* in a different order than OpenCV enumerates them (Continuity
 * cameras renumber, a running robot claiming an index shifts the probe
 * list). So this gallery leads with the only honest identity — a live
 * snapshot per index — and shows the name roster separately, labeled as
 * a roster rather than pretending name[i] belongs to index i.
 */
export default function CameraGallery({ cameras, names }: { cameras: CameraInfo[]; names: CameraName[] }) {
  const [previews, setPreviews] = useState<Record<number, string>>({})
  const [errors, setErrors] = useState<Record<number, string>>({})
  const [loading, setLoading] = useState<Record<number, boolean>>({})
  const urlsRef = useRef<Record<number, string>>({})

  // Object URLs leak real memory until revoked; one place owns their lifetime.
  useEffect(() => () => {
    for (const url of Object.values(urlsRef.current)) URL.revokeObjectURL(url)
  }, [])

  const snap = async (index: number) => {
    setLoading(l => ({ ...l, [index]: true }))
    setErrors(({ [index]: _gone, ...rest }) => rest)
    try {
      const url = await apiBlob(`/api/devices/camera/${index}/preview`)
      if (urlsRef.current[index]) URL.revokeObjectURL(urlsRef.current[index])
      urlsRef.current[index] = url
      setPreviews(p => ({ ...p, [index]: url }))
    } catch (e: any) {
      setErrors(er => ({ ...er, [index]: e?.message ?? String(e) }))
    } finally {
      setLoading(l => ({ ...l, [index]: false }))
    }
  }

  return (
    <div className="camgallery">
      <div className="camgrid">
        {cameras.length === 0 && (
          <p className="hint">No cameras probed — plug one in and rescan.</p>
        )}
        {cameras.map(c => (
          <div key={c.index} className={`camcard${c.claimed_by ? ' claimed' : ''}`}>
            <div className="camcard-head">
              <b>index {c.index}</b>
              <span className="meta">
                {c.width ? `${c.width}×${c.height}` : ''}
                {c.fps ? ` @ ${c.fps}fps` : ''}
              </span>
            </div>
            {c.claimed_by ? (
              <div className="camcard-body claimed-note">
                streaming for <b>{c.claimed_by}</b> — watch it on that robot's card
              </div>
            ) : (
              <div className="camcard-body">
                {previews[c.index]
                  ? <img src={previews[c.index]} alt={`camera index ${c.index} snapshot`}
                         onClick={() => void snap(c.index)} title="click to re-snap" />
                  : (
                    <button className="btn ghost campreview-btn" disabled={!!loading[c.index]}
                            onClick={() => void snap(c.index)}>
                      {loading[c.index] ? 'snapping…' : '📷 snap a preview'}
                    </button>
                  )}
                {errors[c.index] && <div className="camerr">⚠ {errors[c.index]}</div>}
              </div>
            )}
          </div>
        ))}
      </div>
      {names.length > 0 && (
        <>
          <div className="camnames">
            {names.map(n => (
              <span key={n.listing_index} className="chip" title="position in the OS device listing">
                {n.name}
              </span>
            ))}
          </div>
          <p className="hint">
            Attached cameras by name, in OS listing order — which is <em>not</em> OpenCV index
            order. The snapshot is the identity: if you're unsure which index is which, look.
          </p>
        </>
      )}
    </div>
  )
}
